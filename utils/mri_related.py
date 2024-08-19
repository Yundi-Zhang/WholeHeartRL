import torch
import numpy as np
from numpy.lib.stride_tricks import as_strided


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def cartesian_mask_yt(ky, dim_t, acc, sample_n=5, centred=False, uniform=True):
    mask_stack = []
    for t in range(dim_t):
        mask = cartesian_mask((1, ky, 1), acc, sample_n=sample_n, centred=centred, uniform=uniform)
        mask_stack.append(mask)
    mask_stack = np.concatenate(mask_stack, axis=0)
    return mask_stack.squeeze()

def cartesian_mask_yt_uniform(ky, dim_t, acc):
    mask_stack = []
    for t in range(dim_t):
        idx = np.random.choice(ky, int(ky/acc), False)
        mask = np.zeros((1, ky, 1))
        mask[0, idx, 0] = 1
        mask_stack.append(mask)
    mask_stack = np.concatenate(mask_stack, axis=0)
    return mask_stack.squeeze()

def cartesian_mask(shape, acc, sample_n=10, centred=False, uniform=False):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn"t have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n
    if uniform:
        pdf_x = np.ones(Nx) / (Nx - sample_n)
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0


    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, p=pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = np.fft.ifftshift(mask, axes=(-1, -2))

    return mask


def fft2(x, dim=(-2,-1)):
    return torch.fft.fft2(x, dim=dim, norm="ortho")


def ifft2(X, dim=(-2,-1)):
    return torch.fft.ifft2(X, dim=dim, norm="ortho")


def fft2c(x, dim=(-2,-1)):
    return torch.fft.fftshift(fft2(torch.fft.ifftshift(x, dim), dim), dim)


def ifft2c(x, dim=(-2,-1)):
    return torch.fft.fftshift(ifft2(torch.fft.ifftshift(x, dim), dim), dim)
