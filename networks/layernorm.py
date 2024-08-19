import numpy as np
import torch
from torch import nn
import unittest


class ComplexNormalizationBase(nn.Module):
    def __init__(self,
                 channel_last=True,
                 epsilon=1e-5,
                 affine=True):
        super().__init__()
        self.epsilon = epsilon
        self.channel_last = channel_last
        self.reduction_axes = []
        self.affine = affine

    def cal_reduction_axes(self, input_shape):
        ndim = len(input_shape)
        if self.channel_last:
            reduction_axes = list(range(1, ndim-1))
        else:
            reduction_axes = list(range(2, ndim))
        self.reduction_axes += reduction_axes

    def forward(self, x):
        return self.whiten2x2(x)

    def whiten2x2(self, x):
        # 1. compute mean regarding the specified axes
        mu = x.mean(dim=self.reduction_axes, keepdim=True)
        x = x - mu

        xre = torch.real(x)
        xim = torch.imag(x)

        # 2. construct 2x2 covariance matrix
        # Stabilize by a small epsilon.
        cov_uu = xre.var(dim=self.reduction_axes, unbiased=False, keepdim=True) + self.epsilon
        cov_vv = xim.var(dim=self.reduction_axes, unbiased=False, keepdim=True) + self.epsilon
        cov_vu = cov_uv = (xre * xim).mean(dim=self.reduction_axes, keepdim=True)

        # 3. get R = [[p, q], [r, s]], with E R c c^T R^T = R M R = I
        sqrdet = torch.sqrt(cov_uu * cov_vv - cov_uv * cov_vu)
        denom = sqrdet * torch.sqrt(cov_uu + 2 * sqrdet + cov_vv)

        p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom
        r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom

        # 4. apply R to x (manually)
        re = xre * p + xim * r
        im = xre * q + xim * s

        if self.affine:
            # todo: verify this! @kerstin
            # todo: currently it just apply for layernorm, not for instance_norm
            out_re = re * self.weight[..., 0] + im * self.weight[..., 2] + self.bias[..., 0]
            out_im = re * self.weight[..., 2] + im * self.weight[..., 1] + self.bias[..., 1]

        out_re = re.float()
        out_im = im.float()

        return torch.complex(out_re, out_im)


class ComplexLayerNormalization(ComplexNormalizationBase):
    def __init__(self, normalized_shape, affine=True, epsilon=1e-5):
        super().__init__(epsilon=epsilon, affine=affine)
        self.normalized_shape = [normalized_shape] if isinstance(normalized_shape, int) else normalized_shape
        self.reduction_axes = list(range(-len(self.normalized_shape), 0))

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(*self.normalized_shape, 3))  # uu, vv, uv
            self.bias = nn.Parameter(torch.Tensor(*self.normalized_shape, 2))
        else:
            self.register_parameter("weight", None)  # todo: need to do this? haven"t understood
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.weight[..., :2], 1/1.4142135623730951)  # todo: in repo complexPyTorch it"s 1.41 not 1/1.41, which one is right?
            nn.init.zeros_(self.weight[..., 2])
            nn.init.zeros_(self.bias)


    def forward(self, x):
        for i, dim in enumerate(self.reduction_axes):
            assert x.shape[dim] == self.normalized_shape[i], "Embedding size mismatched!"
        return self.whiten2x2(x)


class ComplexNormTest(unittest.TestCase):
    def _test_norm(self, shape, layer_norm=False, normalized_shape=32, affine=False):

        if layer_norm:
            model = ComplexLayerNormalization(normalized_shape=normalized_shape, affine=affine)
        else:
            pass

        x = torch.complex(torch.randn(shape), torch.randn(shape)*2)
        xn = model(x)

        if layer_norm:
            normalized_shape = [normalized_shape] if isinstance(normalized_shape, int) else normalized_shape
            axes = list(range(-1, -len(normalized_shape)-1, -1))

        xnre = torch.real(xn)
        xnim = torch.imag(xn)

        xre = torch.real(x)
        xim = torch.imag(x)

        # # compare to the original distribution
        np_mu_o = torch.mean(x, axes).numpy()
        uu_o = torch.var(xre, axes).numpy()
        vv_o = torch.var(xim, axes).numpy()
        uv_o = torch.mean(xre * xnim, axes).numpy()


        np_mu = torch.mean(xn, axes).numpy()
        self.assertTrue((np.mean(np_mu) < 1e-6).all())

        uu = torch.var(xnre, axes).numpy()
        vv = torch.var(xnim, axes).numpy()
        uv = torch.mean(xnre * xnim, axes).numpy()

        self.assertTrue((np.mean(uu - 1) < 2e-3).all())
        self.assertTrue((np.mean(vv - 1) < 2e-3).all())
        self.assertTrue((np.mean(uv) < 2e-5).all())

    def test1_layer(self):
        # VIT shape
        self._test_norm([16, 64, 768], layer_norm=True, normalized_shape=768, affine=False)

    def test2_layer(self):
        # Swin shape
        self._test_norm([16, 4096, 32], layer_norm=True, normalized_shape=32, affine=False)

    def test3_layer(self):
        # VIT shape
        self._test_norm([16, 64, 768], layer_norm=True, normalized_shape=[64, 768], affine=False)

    def test4_layer(self):
        # Swin shape
        self._test_norm([16, 32, 256, 256], layer_norm=True, normalized_shape=[256, 256], affine=False)


if __name__ == "__main__":
    # unittest.test()
    test = ComplexNormTest()
    test.test2_instance()
    #print("hello")
