import numpy as np
import torch
from torch import nn
from timm.models.vision_transformer import Block
from models.reconstruction_models import BasicModule
from networks.decoders import UNETR_decoder
from networks.losses import SegmentationCriterion
from utils.general import to_1hot
from utils.logging_related import imgs_to_wandb_video
from utils.model_related import sincos_pos_embed, PatchEmbed


class SegMAE(BasicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.enc_embed_dim = kwargs.get("enc_embed_dim")
        self.dec_embed_dim = kwargs.get("dec_embed_dim")
        self.patch_embed_cls = globals()[kwargs.get("patch_embed_cls")]
        val_dataset= kwargs.get("val_dset")
        
        self.num_classes = val_dataset.num_classes
        self.img_shape = val_dataset[0][0].shape
        self.data_view = val_dataset.view
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
        # Segmentation decoder and head
        S = self.img_shape[0]
        self.decoder_embed = nn.Linear(self.enc_embed_dim, self.dec_embed_dim//S, bias=True)
        self.decoder = UNETR_decoder(in_channels=S,
                                     out_channels=self.num_classes,
                                     img_size=self.img_shape,
                                     patch_size=kwargs.get("patch_size"),
                                     feature_size=kwargs.get("feature_size"),
                                     upsample_kernel_sizes=kwargs.get("upsample_kernel_sizes"),
                                     hidden_size=self.dec_embed_dim,
                                     spatial_dims=kwargs.get("spatial_dims"))
        self.segmentation_criterion = SegmentationCriterion(**kwargs)
        self.save_hyperparameters()
    
    def forward_encoder(self, imgs):
        # Embed patches: (B, S, T, H, W) -> (B, S * num_patches, embed_dim)
        x = self.patch_embed(imgs)

        # Add positional embedding: (B, S * num_patches, embed_dim)
        enc_pos_embed = self.enc_pos_embed.repeat(x.shape[0], 1, 1)
        x = x + enc_pos_embed[:, 1:, :] # No class token for semantic segmentation
        
        # Append cls token: (B, 1 + length * mask_ratio, embed_dim)
        cls_token = self.cls_token + enc_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply MAE encoder
        hidden_latents = []
        for blk in self.encoder:
            hidden_latents.append(x[:, 1:, :])
            x = blk(x)
        encoder_output = self.encoder_norm(x)
        return encoder_output, hidden_latents
    
    def forward_decoder(self, imgs, encoder_output, hidden_latents):
        # Apply UNETR decoder
        decoder_imgs = imgs
        decoder_x = self.decoder_embed(encoder_output)[:, 1:, :]
        decoder_hidden_states = []
        for i in range(len(hidden_latents)):
            if i not in [2, 4]:
                continue
            else:
                decoder_hidden_states.append(self.decoder_embed(hidden_latents[i]))
        decoder_output = self.decoder(x_in=decoder_imgs, x=decoder_x, hidden_states_out=decoder_hidden_states)
        preds = torch.nn.functional.softmax(decoder_output, dim=1)
        return preds
    
    def forward(self, imgs):
        encoder_output, hidden_latents = self.forward_encoder(imgs)
        pred_segs = self.forward_decoder(imgs, encoder_output, hidden_latents)
        return pred_segs
    
    def training_step(self, batch, batch_idx, mode="train"):
        imgs, segs, sub_idx = batch
        
        pred_segs_ = self.forward(imgs)
        gt_segs_ = to_1hot(segs, num_class=self.num_classes) # (B, S, T, H, W, class)
        gt_segs_ = gt_segs_.moveaxis(-1, 1) # (B, class, S, T, H, W)
        loss, dice_score = self.segmentation_criterion(pred_segs_, gt_segs_)
        
        # Logging metrics and median
        self.log_seg_metrics(loss, dice_score, mode=mode)
        if mode == "train" or mode == "val":
            if mode == "val":
                self.log_dict({f"{mode}_Dice_FG": dice_score[1:].mean().detach().item()}) # For checkpoint tracking
                
            log_rate = eval(f"self.{mode}_rate")
            if self.current_epoch > 0 and ((self.current_epoch + 1) % log_rate == 0):
                if (sub_idx == 0).any():
                    i = (sub_idx == 0).argwhere().squeeze().item()
                    pred_seg = torch.argmax(pred_segs_[i], dim=0).detach()
                    self.log_seg_videos(imgs[i], segs[i], pred_seg, sub_idx[i], mode=mode)
                if (sub_idx == 1).any():
                    i = (sub_idx == 1).argwhere().squeeze().item()
                    pred_seg = torch.argmax(pred_segs_[i], dim=0).detach()
                    self.log_seg_videos(imgs[i], segs[i], pred_seg, sub_idx[i], mode=mode)
        return loss
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _ = self.training_step(batch, batch_idx, mode="val")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if not self.vis:
            _ = self.training_step(batch, batch_idx, mode="test")
        else:
            self.visualization(batch, batch_idx, mode="vis")
    
    def log_seg_metrics(self, loss, dice_score, mode="train"):
        self.module_logger.update_metric_item(f"{mode}/seg_loss", loss.item(), mode=mode)
        self.module_logger.update_metric_item(f"{mode}/Dice_FG", dice_score[1:].mean().detach().item(), mode=mode)
        self.module_logger.update_metric_item(f"{mode}/Dice", dice_score.mean().detach().item(), mode=mode)
        self.module_logger.update_metric_item(f"{mode}/Dice_BG", dice_score[0].detach().item(), mode=mode)
        if self.data_view == 0 or self.data_view == 2: # For both axes and only short-axis views
            self.module_logger.update_metric_item(f"{mode}/Dice_LVBP", dice_score[1].detach().item(), mode=mode)
            self.module_logger.update_metric_item(f"{mode}/Dice_LVMYO", dice_score[2].detach().item(), mode=mode)
            self.module_logger.update_metric_item(f"{mode}/Dice_RVBP", dice_score[3].detach().item(), mode=mode)
        if self.data_view == 1 or self.data_view == 2: # For both axes and only long-axis views
            self.module_logger.update_metric_item(f"{mode}/Dice_LA", dice_score[-2].detach().item(), mode=mode)
            self.module_logger.update_metric_item(f"{mode}/Dice_RA", dice_score[-1].detach().item(), mode=mode)
    
    def log_seg_videos(self, imgs, segs, pred_segs, sub_idx, mode="train"):
        sub_path = eval(f"self.trainer.datamodule.{mode}_dset").subject_paths[sub_idx]
        sub_id = sub_path.parent.name
        
        # Overlay the segmentations on the groud truth images
        gt_imgs = imgs[:, :, None, ...] # (S, T, H, W)
        gt_imgs = torch.tile(gt_imgs, dims=(1, 1, 3, 1, 1))
        color_pred = torch.zeros_like(gt_imgs)
        color_gt = torch.zeros_like(gt_imgs)
        colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
        for u in torch.unique(segs):
            if u == 0: continue
            color = colors[u]
            for j in range(3):
                color_pred[:, :, j, ...][pred_segs == u] = color[j]
                color_gt[:, :, j, ...][segs == u] = color[j]
        
        pred_overlay = torch.where(color_pred > 0, color_pred, gt_imgs)
        gt_overlay = torch.where(color_gt > 0, color_gt, gt_imgs)
        
        cat_slices = []
        for s in range(gt_imgs.shape[0]):
            if self.data_view != 0 and s == 1: continue
            cat_slices.append(torch.cat([gt_imgs[s], gt_overlay[s], pred_overlay[s]], dim=2))
            
        cat_all = torch.cat([cat_slices[p] for p in range(len(cat_slices))], dim=3) # (T, 3, H_, W_)
        cat_all_log = imgs_to_wandb_video(cat_all, prep=True, in_channel=3)
        self.module_logger.update_video_item(f"{mode}_video/pred_segs", sub_id, cat_all_log, mode=mode)
