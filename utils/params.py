"""Parameters for the module in a nested dataclass manner."""

from dataclasses import asdict, dataclass, fields
import yaml
from typing import Any, Dict, Optional, Tuple


@dataclass
class GeneralParams:
    # wandb logging configs
    wandb_group_name: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_entity: str = "walala"
    wandb_disabled: str = "true"
    
    seed: int = 1
    gpu: int = 0
    freeze_encoder: bool = False
    resume_training: bool = False
    load_encoder: bool = False
    ckpt_path: str = None

    
@dataclass
class DataParams:
    replace_processed: bool = False
    augment: bool = True
    path_file_name: str = "data_paths.pkl"
    selected_subject_pkl_name: str = "selected_subject.pkl"
    target_pkl_name: str = "target_table.pkl"
    
    num_train: int = 6000
    train_num_per_epoch: int = 1000
    num_val: int = 100
    num_test: int = 100
    batch_size: int = 1
    num_workers: int = 4
    
    dataset_cls: str = "Cardiac3DplusTAllAX"
    load_seg: bool = False
    all_value_names: Tuple[str, ...] = ("Age", "LVEDV (mL)", "LVESV (mL)", "LVSV (mL)", "LVEF (%)", "LVCO (L/min)", "LVM (g)", "RVEDV (mL)", "RVESV (mL)", "RVSV (mL)", "RVEF (%)", "LAV max (mL)", "LAV min (mL)", "LASV (mL)", "LAEF (%)", "RAV max (mL)", "RAV min (mL)", "RASV (mL)", "RAEF (%)")
    target_value_name: str = "LVM (g)" # Only select the ones that are involved in training 
    
    sax_slice_num: int = 6
    time_frame: int = 50
    image_size: Tuple[int, ...] = (128, 128)
    

@dataclass
class TrainerParams:
    accelerator: str = "gpu"
    max_epochs: int = 10_000
    check_val_every_n_epoch: int = 5
    # devices: int = 1


@dataclass
class ReconMAEParams:
    enc_embed_dim: int = 1025 # has to be divisible by 8 or 6 for one modality. When it"s for two modalities, add one more dimension for distinguishing two modalities.
    enc_depth: int = 6
    enc_num_heads: int = 5
    mlp_ratio: float = 4.
    dec_embed_dim: int = 1025 # has to be divisible by 8 or 6 for one modality. When it"s for two modalities, add one more dimension for distinguishing two modalities.
    dec_depth: int = 2
    dec_num_heads: int = 5
    

@dataclass
class SegMAEParams:
    enc_embed_dim: int = 1025
    enc_depth: int = 6
    enc_num_heads: int = 5
    mlp_ratio: float = 4.
    feature_size: int = 16
    dec_embed_dim: int = 1152
    spatial_dims: int = 3
    upsample_kernel_sizes: Tuple[Tuple, ...] = ((1, 2, 2), (5, 2, 2), (5, 2, 2))
    

@dataclass
class RegrMAEParams:
    enc_embed_dim: int = 1025
    enc_depth: int = 6
    enc_num_heads: int = 5
    mlp_ratio: float = 4.
    dec_embed_dim: int = 256
    dec_depth: int = 2
    regressor_type: str = "cls_token"
    

@dataclass
class TrainingParams:
    train_log_rate: int = 5
    val_log_rate: int = 10
    
    # Patchify
    patch_embed_cls: str = "PatchEmbed"
    patch_size: Tuple[int, ...] = (5, 8, 8)
    patch_in_channels: int = 1
    mask_type: str = "random"
    mask_ratio: float = 0.7
    
    # Optimizer and scheduler
    dropout: float = 0.0 # TODO
    lr: float = 1e-4
    min_lr: float = 0.0
    warmup_epochs: int = 20
    optim_weight_decay: float = 0.05
    
    # Loss
    loss_types: Tuple[str, ...] = ("mse")
    loss_weights: Tuple[float, ...] = (1.0,)
    
    
@dataclass
class ModuleParams:
    task_idx: Optional[int] = None
    module_idx: Optional[int] = None
    
    training_params: TrainingParams = None
    recon_hparams: ReconMAEParams = None
    seg_hparams: SegMAEParams = None
    regr_hparams: RegrMAEParams = None
    
    
@dataclass
class Params:
    general: GeneralParams = None
    data: DataParams = None
    trainer: TrainerParams = None
    module: ModuleParams = None
    
    
def load_config_from_yaml(file_path):
    config_data = dict()
    if file_path is not None:
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)

    # Get the default values from the data class
    params = Params(general=GeneralParams(), 
                    data=DataParams(), 
                    trainer=TrainerParams(), 
                    module=ModuleParams(recon_hparams=ReconMAEParams(),
                                        seg_hparams=SegMAEParams(),
                                        regr_hparams=RegrMAEParams(),
                                        training_params=TrainingParams()))
    update_params = update_dataclass_from_dict(params, config_data)

    return update_params


def update_dataclass_from_dict(params, config_data: Dict[str, Any]):
    updated_fields = {}
    instance_dict = asdict(params)
    for key in config_data:
        if is_field_name(params, key):
            value = config_data[key]
            if isinstance(value, dict) and hasattr(getattr(params, key), '__dataclass_fields__'):
                # Recursively update nested dataclass
                updated_value = update_dataclass_from_dict(getattr(params, key), value)
                updated_fields[key] = updated_value
            else:
                updated_fields[key] = value
            instance_dict.update(updated_fields)
        else:
            raise NameError(f"{key} is not defined in the dataclass")
    return params.__class__(**instance_dict)


def is_field_name(dataclass_type, field_name):
    return field_name in [f.name for f in fields(dataclass_type)]