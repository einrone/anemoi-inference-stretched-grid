import os 
import logging 

from typing import Any
from functools import cached_property
from pathlib import Path 

from anemoi.utils.checkpoints import load_metadata
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)

class Checkpoint:
    AIFS_BASE_SEED = None
    def __init__(
            self, 
            ckpt: str,
            **kwargs: Any
            ) -> str:
        assert os.path.exists(ckpt), f"The given checkpoint does not exist!"
        self.ckpt = ckpt 

    @cached_property
    def _metadata(self) -> dict:
        try:
            print(DotDict(load_metadata(self.ckpt)).training)
            return DotDict(load_metadata(self.ckpt))
        except Exception as e:
            LOGGER.warning("Could not load and peek into the checkpoint metadata. Raising an expection")
            raise e

    @cached_property
    def base_seed(self) -> None:
        os.environ["AIFS_BASE_SEED"] = f"{self._metadata.seed}"

        self.AIFS_BASE_SEED = os.get(os.environ("AIFS_BASE_SEED"), None)
        if self.AIFS_BASE_SEED:
            LOGGER.info(f"AIFS_BASE_SEED set to: {self.AIFS_BASE_SEED}")
            return self.AIFS_BASE_SEED
        
        self.AIFS_BASE_SEED = 1234
        LOGGER.info(f"Could not find AIFS_BASE_SEED. Setting to a random number {self.AIFS_BASE_SEED}")
        return self.AIFS_BASE_SEED
    
    @cached_property
    def data_indices(self) -> dict:
        return self._metadata.data_indices.data
    
    @cached_property
    def model_indices(self) -> dict:
        return self._metadata.data_indices.model 
    
    @cached_property
    def num_gridpoints(self):
        return self._metadata.dataset.shape[-1]

    @cached_property
    def num_gridpoints_lam(self):
        return self._metadata.dataset.specific.forward.forward.forward.datasets[0].shape[-1]
    
    @cached_property
    def num_gridpoints_global(self):
        return self._metadata.dataset.specific.forward.forward.forward.datasets[1].shape[-1]
    @cached_property
    def num_features(self):
        return len(self.model_indices.input.full)

    @cached_property
    def config(self) -> dict:
        return self._metadata.config

    @cached_property
    def graph_config(self) -> dict:
        return self.config.graphs
    
    @cached_property
    def name_to_index(self) -> dict:
        return {name : index for index,name in enumerate(self._metadata.dataset.variables)}

    @cached_property
    def index_to_name(self) -> dict:
        return {index : name for index,name in enumerate(self._metadata.dataset.variables)}

    def _make_indices_mapping(self, indices_from, indices_to):
        assert len(indices_from) == len(indices_to)
        return {i: j for i, j in zip(indices_from, indices_to)}
    
    @cached_property
    def model_output_index_to_name(self) -> dict:
        """Return the mapping between output tensor index and variable name"""
        mapping = self._make_indices_mapping(
            self.model_indices.output.full,
            self.data_indices.output.full,
        )
        return {k: self._metadata.dataset.variables[v] for k, v in mapping.items()}
    
    @cached_property
    def model_output_name_to_index(self) -> dict:
        return {name : index for index, name in self.model_output_index_to_name.items()}
    
    @cached_property
    def lam_yx_dimensions(self) -> tuple:
        return self._metadata.dataset.specific.forward.forward.forward.datasets[0].forward.forward.attrs.field_shape
    @cached_property
    def era_to_meps(self) -> tuple:
        return self._metadata.dataset.specific.forward.forward.forward.datasets[0].forward.forward.attrs.era_to_meps_mapping

    @cached_property
    def LAM_latlon(self) -> tuple | list:
        return self._metadata.dataset.specific.forward.forward.forward.datasets[0]

    @cached_property
    def latitudes(self) -> Any:
        raise NotImplementedError
    
if __name__ == "__main__":
    import torch 

    #ckpt = "/lustre/storeB/project/nwp/aifs/aram/fix-memory-issue/experiments2/legendary_gnome/r4/checkpoints/inference-aifs-by_step-epoch_000-step_000150.ckpt"
    ckpt = "/lustre/storeB/project/nwp/bris/aram/fix-memory-issue/experiments2/inference-aifs-by_step-epoch_000-step_000150.ckpt"
    #ckpt = "/lustre/storeB/project/nwp/bris/experiments/lousy_algorithm/checkpoints/inference-aifs-by_epoch-epoch_603-val_wmse_7.360e-03.ckpt"
    #name_to_index = Checkpoint(ckpt).model_output_name_to_index
    #ckpt = "/home/arams/Documents/project/anemoi-inference-stretched-grid/ckpt/inference-anemoi-by_epoch-epoch_653-step_068016.ckpt"
    #select = [name_to_index[name] for  name in ["2t", "10u", "10v", "msl", "tp", "2d"]]
    #print(select)
    #ckpt = "/home/arams/Documents/project/anemoi-inference-stretched-grid/ckpt/inference-anemoi-by_epoch-epoch_011-step_060000.ckpt"
    print(Checkpoint(ckpt)._metadata.keys())
    #print("####################################################")

    #graph_config = Checkpoint(ckpt).graph_config

    #checkpoint = torch.load(ckpt, map_location="cpu")
    #print(checkpoint.hparams)
    #print(checkpoint.data_indices)
    #print(checkpoint.statistics)
    #spatial_mask = {}
    #for mesh_name, mesh in checkpoint.graph_data.items():
    #    if isinstance(mesh_name, str) and mesh_name != graph_config.hidden_mesh.name:
    #        spatial_mask[mesh_name] = mesh.get("dataset_idx", None)
    #print(spatial_mask)

    #print(Checkpoint(ckpt)._metadata)
