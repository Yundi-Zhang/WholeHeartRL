import argparse
from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path

import torch
import wandb

from data.dataloaders import CMRDataModule
from models.reconstruction_models import ReconMAE
from models.regression_models import RegrMAE, ResNet18Module, ResNet50Module
from models.segmentation_models import SegMAE
from utils.data_related import get_data_paths
from utils.params import load_config_from_yaml

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, BaseFinetuning


def parser_command_line():
    "Define the arguments required for the script"
    parser = argparse.ArgumentParser(description="Masked Autoencoder Downstream Tasks",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser = parser.add_subparsers(dest="pipeline", help="pipeline to run")
    # Arguments for training
    parser_train = subparser.add_parser("train", help="train the model")
    parser_train.add_argument("-c", "--config", help="config file (.yml) containing the ¢hyper-parameters for inference.")
    parser_train.add_argument("-g", "--wandb_group_name", default=None, help="specify the name of the group")
    parser_train.add_argument("-n", "--wandb_run_name", default=None, help="specify the name of the experiment")
    # Arguments for evaluation
    parser_eval = subparser.add_parser("eval", help="evaluate the model")
    parser_eval.add_argument("-c", "--config", help="config file (.yml) containing the hyper-parameters for inference.")
    parser_eval.add_argument("-g", "--wandb_group_name", default=None, help="specify the name of the group")
    parser_eval.add_argument("-n", "--wandb_run_name", default=None, help="specify the name of the experiment")
    return parser.parse_args()


def main():
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision("medium")
    
    args = parser_command_line() # Load the arguments from the command line
    try:
        config_path = args.config
    except AttributeError:
        config_path = None
    params = load_config_from_yaml(config_path)
    paths = get_data_paths()
    os.environ["WANDB_DISABLED"] = params.general.wandb_disabled
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Configure accelerator and devices
    seed_everything(params.general.seed, workers=True) # Sets seeds for numpy, torch and python.random.

    # Initialize wandb logging
    wandb_kwargs = dict()
    wandb_kwargs["entity"] = params.general.wandb_entity
    wandb_kwargs["group"] = args.wandb_group_name if args.wandb_group_name is not None else params.general.wandb_group_name
    wandb_run_name = args.wandb_run_name if args.wandb_run_name is not None else params.general.wandb_run_name
    time_now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    wandb_kwargs["name"] = f"{wandb_run_name}_{time_now}"
    wandb_kwargs["resume"] = "allow" # Resume if the run_id is provided and identical to a previous run otherwise, start a new run
    if params.general.wandb_run_id is not None:
        wandb_kwargs["id"] = params.general.wandb_run_id
    logger = CSVLogger(paths.log_folder)
    wandb.init(project="MAE", config=asdict(params), **wandb_kwargs,)
    
    # Initialize data module
    table_condition_dict = {"healthy_cases": True, "sorting_with_age": True,}
    # CMR_condition_dict = {"min_slice_num": 9, "min_image_size": [128, 128]} # TODO
    data_module = CMRDataModule(load_dir=paths.dataset_folder, 
                                processed_dir=paths.processed_folder,
                                all_feature_tabular_dir=paths.all_feature_tabular_dir, 
                                biomarker_tabular_dir=paths.biomarker_tabular_dir,
                                dataloader_file_folder=paths.dataloader_file_folder,
                                extra_tabular_dir=paths.extra_tabular_dir,
                                cmr_path_pickle_name=paths.cmr_path_pickle_name,
                                biomarker_table_pickle_name=paths.biomarker_table_pickle_name,
                                processed_table_pickle_name=paths.processed_table_pickle_name,
                                table_condition_dict=table_condition_dict,
                                
                                **params.data.__dict__)
    data_module.setup("fit")

    # Initialze lighting module
    module_LUT = {"reconstruction": [ReconMAE],
                  "regression": [RegrMAE, ResNet18Module, ResNet50Module],
                  "segmentation": [SegMAE]}
    if params.module.task_idx == 0:
        module_cls = module_LUT["reconstruction"][params.module.module_idx]
        module_params = {**params.module.training_params.__dict__, **params.module.recon_hparams.__dict__}
    elif params.module.task_idx == 1:
        module_cls = module_LUT["regression"][params.module.module_idx]
        module_params = {**params.module.training_params.__dict__, **params.module.regr_hparams.__dict__}
    elif params.module.task_idx == 2:
        module_cls = module_LUT["segmentation"][params.module.module_idx]
        module_params = {**params.module.training_params.__dict__, **params.module.seg_hparams.__dict__}
    else:
        raise NotImplementedError
    model = module_cls(val_dset=data_module.val_dset, **module_params)
    
    # Check the resuming and loading of the checkpoints
    resume_ckpt_path = None
    if params.general.resume_training:  # Resume training
        assert params.general.ckpt_path != None, "The path for checkpoint is not provided."
        resume_ckpt_path = params.general.ckpt_path
    
    if params.general.load_encoder: # Load pretraining encoder
        assert params.general.ckpt_path != None, "The path for checkpoint is not provided."
        ckpt = torch.load(params.general.ckpt_path)
        pretrained_dict = ckpt["state_dict"]
        processed_dict = {}
        pretrained_params = ["cls_token", "enc_pos_embed", "mask_token", "patch_embed", "encoder", "encoder_norm"]
        for k in model.state_dict().keys():
            decomposed_k = k.split(".")
            if decomposed_k[0] in pretrained_params:
                processed_dict[k] = pretrained_dict[k]
        model.load_state_dict(processed_dict, strict=False)
    
    if params.general.freeze_encoder: # Freeze encoder
        BaseFinetuning.freeze([model.patch_embed, model.encoder, model.encoder_norm])
                
    # Monitor foreground dice for segmentation. When reconstruction, monitor PSNR. MAE for regression.
    if params.general.resume_training:
        ckpt_dir = Path(resume_ckpt_path).parent
    else:
        ckpt_dir = os.path.join(f"{paths.log_folder}/checkpoints/{wandb_run_name}/{time_now}")
    monitor_LUT = [
        ("val_PSNR", "model-{epoch:03d}-{val_PSNR:.2f}", "max"), # Reconstruction
        ("val_MAE", "model-{epoch:03d}-{val_MAE:.2f}", "min"), # Regression
        ("val_Dice_FG", "model-{epoch:03d}-{val_Dice_FG:.2f}", "max"), # Segmentation
    ]
    monitor_metric, ckpt_filename, monitor_mode = monitor_LUT[params.module.task_idx]
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, filename=ckpt_filename, monitor=monitor_metric, 
                                          mode=monitor_mode, save_top_k=5, save_last=True, verbose=True,)
    
    # Initialize trainer
    trainer = Trainer(
        default_root_dir=paths.log_folder,
        logger=logger,
        callbacks=[checkpoint_callback],
        fast_dev_run=False,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        num_sanity_val_steps=2,
        benchmark=True,

        **params.trainer.__dict__,
    )

    if args.pipeline == "train":
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt_path)
        trainer.test(model, datamodule=data_module)
    elif args.pipeline == "eval":
        trainer.test(model, datamodule=data_module)
    wandb.finish() # Finish logging


if __name__ == "__main__":
    main()
