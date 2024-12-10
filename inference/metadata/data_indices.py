from functools import cached_property

from checkpoint import Checkpoint

class DataIndices(Checkpoint):
    def __init__(self, ckpt):
        super().__init__(ckpt)
    
    @cached_property
    def _data_indices_input(self) -> dict:
        return self.data_indices.input
    
    @cached_property
    def data_indices_full(self) -> dict:
        return self._data_indices_input.full
    
    @cached_property
    def data_indices_forcing(self) -> dict:
        return self._data_indices_input.forcing
    
    @cached_property
    def data_indices_diagnostic(self) -> dict:
        return self._data_indices_input.diagnostic
    
    @cached_property
    def data_indices_prognostic(self) -> dict:
        return self._data_indices_input.prognostic
    

class ModelIndices(Checkpoint):
    def __init__(self, ckpt):
        super().__init__(ckpt)

    @cached_property
    def _model_indices_model(self) -> dict:
        return self.model_indices.input
    
    @cached_property
    def model_indices_full(self) -> dict:
        return self._model_indices_input.full
    
    @cached_property
    def model_indices_diagnostic(self) -> dict:
        return self._model_indices_input.diagnostic
    
    @cached_property
    def model_indices_prognostic(self) -> dict:
        return self._model_indices_input.prognostic
    @cached_property
    def model_indices_forcing(self) -> dict:
        return self._model_indices_input.forcing
    
    @cached_property
    def model_indices_diagnostic(self) -> dict:
        return self._model_indices_input.diagnostic
    
    @cached_property
    def model_indices_prognostic(self) -> dict:
        return self._model_indices_input.prognostic

if __name__ == "__main__":
    ckpt = "/lustre/storeB/project/nwp/aifs/aram/fix-memory-issue/experiments2/legendary_gnome/r4/checkpoints/inference-aifs-by_step-epoch_000-step_000150.ckpt"
    print(DataIndices(ckpt).data_indices_input)
    print(DataIndices(ckpt).data_indices_full)
    print(DataIndices(ckpt).data_indices_prognostic)
    print(DataIndices(ckpt).data_indices_diagnostic)
    print(DataIndices(ckpt).data_indices_forcing)