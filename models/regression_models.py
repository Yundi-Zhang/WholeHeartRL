import torch
from torch import nn
from timm.models.vision_transformer import Block
from torchvision.models import resnet18, resnet50
from models.reconstruction_models import BasicModule
from networks.decoders import LinearDecoder
from networks.losses import RegressionCriterion
from utils.model_related import PatchEmbed, sincos_pos_embed


class RegrMAE(BasicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.enc_embed_dim = kwargs.get("enc_embed_dim")
        self.dec_embed_dim = kwargs.get("dec_embed_dim")
        self.patch_embed_cls = globals()[kwargs.get("patch_embed_cls")]
        val_dataset= kwargs.get("val_dset")
        self.regressor_type = kwargs.get("regressor_type")
        self.img_shape = val_dataset[0][0].shape
        self.use_both_axes = True if val_dataset.get_view() == 2 else False # For positional embedding
        # --------------------------------------------------------------------------
        # MAE encoder
        self.patch_embed = self.patch_embed_cls(self.img_shape, 
                                                in_channels=kwargs.get("patch_in_channels"), 
                                                patch_size=kwargs.get("patch_size"), 
                                                out_channels=kwargs.get("enc_embed_dim"), )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.patch_embed.out_channels))
        self.enc_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, self.patch_embed.out_channels), 
                                      requires_grad=False)
        self.encoder = nn.ModuleList([Block(dim=self.patch_embed.out_channels, 
                                            num_heads=kwargs.get("enc_num_heads"), 
                                            mlp_ratio=kwargs.get("mlp_ratio"), 
                                            qkv_bias=True,)
                                      for i in range(kwargs.get("enc_depth"))])
        self.encoder_norm = nn.LayerNorm(self.patch_embed.out_channels)
        # --------------------------------------------------------------------------
        # Regression decoder and head
        self.dec_embed = nn.Linear(self.enc_embed_dim, self.dec_embed_dim, bias=True)
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, self.dec_embed_dim), 
                                          requires_grad=False) # with cls token
        if self.regressor_type == "linear":
            self.regressor = LinearDecoder(self.dec_embed_dim * self.patch_embed.num_patches, 
                                           self.dec_embed_dim, kwargs.get("dec_depth"))
        elif self.regressor_type == "cls_token":
            self.regressor = nn.AdaptiveAvgPool2d((1, 1))
        self.regression_criterion = RegressionCriterion(**kwargs)
        self.save_hyperparameters()
    
    def initialize_parameters(self):        
        # Initialize (and freeze) pos_embed by sin-cos embedding
        dec_pos_embed = sincos_pos_embed(self.dec_embed_dim, self.patch_embed.grid_size, cls_token=True,
                                         use_both_axes=self.use_both_axes)
        self.dec_pos_embed.data.copy_(dec_pos_embed.unsqueeze(0))
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights) # TODO
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_encoder(self, x):
        """Forward pass of ViT encoder
        input: [B, S, T, H, W] torch.Tensor
        output:
            latent: [B, num_patches, embed_dim] torch.Tensor
        """
        # Embed patches: (B, S, T, H, W) -> (B, S * T * num_patches, embed_dim)
        x = self.patch_embed(x)

        # Add positional embedding: (B, S * T * num_patches, embed_dim)
        enc_pos_embed = self.enc_pos_embed.repeat(x.shape[0], 1, 1)
        x = x + enc_pos_embed[:, 1:, :]
        
        # Append cls token: (B, 1 + length * mask_ratio, embed_dim)
        cls_token = self.cls_token + enc_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer encoder
        for blk in self.encoder:
            x = blk(x)
        x = self.encoder_norm(x)
        return x
    
    def forward_decoder(self, x):
        """Forwrard pass of regression decoder
        input:
            latent: [B, num_patches, embed_dim] torch.Tensor
        output:
            pred: [B, 1] torch.Tensor
        """
        # Embed tokens
        x = self.dec_embed(x)
        
        # Append mask tokens and add positional embedding in schuffled order
        dec_pos_embed = self.dec_pos_embed.repeat(x.shape[0], 1, 1)
        x = x + dec_pos_embed # TODO: Dropput
        if self.regressor_type == "linear":
            x = x[:, 1:, :] # remove cls token
        elif self.regressor_type == "cls_token":
            x = x[:, :1, :] # keep cls token only
        else:
            raise NotImplementedError
        # Regression decoder
        x = self.regressor(x) # apply regressor
        x = x.squeeze(-1) if self.regressor_type == "cls_token" else x
        x = torch.relu(x) #[B, 1]
        return x
    
    def forward(self, imgs):
        latent = self.forward_encoder(imgs)
        x = self.forward_decoder(latent)
        return x
    
    def training_step(self, batch, batch_idx, mode="train"):
        imgs, values, sub_idx = batch
        
        pred_values = self.forward(imgs)
        loss, mae = self.regression_criterion(pred_values, values)
        
        # Logging metrics and median
        self.log_regr_metrics(loss, mae, mode=mode)
        if mode == "val":
            self.log_dict({f"{mode}_MAE": mae}) # For checkpoint tracking # TODO 
        return loss
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _ = self.training_step(batch, batch_idx, mode="val")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        _ = self.training_step(batch, batch_idx, mode="test")
        
    def log_regr_metrics(self, loss, mae, mode="train"):
        self.module_logger.update_metric_item(f"{mode}/regr_loss", loss.detach().item(), mode=mode)
        self.module_logger.update_metric_item(f"{mode}/mae", mae, mode=mode)


class ResNet18Module(BasicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        val_dataset= kwargs.get("val_dset")
        self.img_shape = val_dataset[0][0].shape
        self.network = self._initial_network()
        self.regression_criterion = RegressionCriterion(**kwargs)
        self.save_hyperparameters()
    
    def _initial_network(self):
        S, T = self.img_shape[:2]
        _network = resnet18(pretrained=False, num_classes=1)
        _network.conv1 = torch.nn.Conv2d(in_channels=S*T, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        return _network
    
    def forward(self, x):
        input_shape = x.shape # [B, S, T, H, W]
        x = x.view(input_shape[0], -1, input_shape[-2], input_shape[-1]) # [B, S*T, H, W]
        x = self.network(x)
        x = torch.relu(x) # [B, 1]
        return x
    
    def training_step(self, batch, batch_idx, mode="train"):
        imgs, values, sub_idx = batch
        
        pred_values = self.forward(imgs)
        loss, mae = self.regression_criterion(pred_values, values)
        
        # Logging metrics and median
        self.log_regr_metrics(loss, mae, mode=mode)
        if mode == "val":
            self.log_dict({f"{mode}_MAE": mae}) # For checkpoint tracking # TODO 
        return loss
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _ = self.training_step(batch, batch_idx, mode="val")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        _ = self.training_step(batch, batch_idx, mode="test")
        
    def log_regr_metrics(self, loss, mae, mode="train"):
        self.module_logger.update_metric_item(f"{mode}/regr_loss", loss.detach().item(), mode=mode)
        self.module_logger.update_metric_item(f"{mode}/mae", mae, mode=mode)
        

class ResNet50Module(ResNet18Module):
    
    def _initial_network(self):
        S, T = self.img_shape[:2]
        _network = resnet50(pretrained=False, num_classes=1)
        _network.conv1 = torch.nn.Conv2d(in_channels=S*T, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        return _network