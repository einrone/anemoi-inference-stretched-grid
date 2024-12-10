import os
import math 
import logging 
from abc import abstractmethod
from typing import Optional, Any

import torch 
import pytorch_lightning as pl

LOGGER = logging.getLogger(__name__)

class AnemoiModel(pl.LightningModule):
    def __init__(
            self, 
            *args: Any, 
            num_device_per_model: Optional[int] = 1,
            num_devices_per_nodes: Optional[int] = 1,
            num_nodes: Optional[int] = 1,
            **kwargs : Any
            ):
        
        # generic abstract modelwrapper which contains everything 
        # for a given model 

        super().__init__(*args, **kwargs)

        assert num_device_per_model >= 1, f"Number of device per model is not greater or equal to 1. Got: {num_device_per_model}"
        assert num_devices_per_nodes >= 1, f"Number of device per node is not greater or equal to 1. Got: {num_device_per_model}"
        assert num_nodes >= 1, f"Number of nodes is not greater or equal to 1. Got : {num_nodes}" 


        self.model_comm_group_id = int(os.environ.get("SLURM_PROCID", "0")) // num_device_per_model
        self.model_comm_group_rank = int(os.environ.get("SLURM_PROCID", "0")) % num_device_per_model
        self.model_comm_num_groups = math.ceil(
            num_devices_per_nodes * num_nodes / num_device_per_model
        )

    def set_model_comm_group(self, model_comm_group) -> None:
        LOGGER.debug("set_model_comm_group: %s", model_comm_group)
        self.model_comm_group = model_comm_group

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass 

    @abstractmethod
    def advance_input_predict(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        pass 

    @abstractmethod
    def predict_step(self, batch: torch.Tensor, batch_idx : int) -> torch.Tensor:
        pass 

class MyModel(AnemoiModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # your model to define
    
    def set_model_comm_group(self, model_comm_group):
        return super().set_model_comm_group(model_comm_group)
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self(x, self.model_comm_group)
    
    @torch.inference_mode
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        pass
    
