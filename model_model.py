import torch
import torch.nn as nn

class BFModule(nn.Module):
    """
    This module is a base class for all modules that need to be compatible with this project.
    It ensures that the module stores its current device and dtype.
    """
    def __init__(self):
        super().__init__()
        self._device = None
        self._dtype = None
    def to(self, *args, **kwargs):
        output = super().to(*args, **kwargs)
        # Extract device and dtype from args/kwargs
        device = next((torch.device(arg) for arg in args if isinstance(arg, (torch.device, str))), 
                     kwargs.get('device', None))
        dtype = next((arg for arg in args if isinstance(arg, torch.dtype)),
                    kwargs.get('dtype', None))
        if device is not None: self._device = device 
        if dtype is not None: self._dtype = dtype
        return output
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device
    @property 
    def dtype(self):
        if self._dtype is None:
            self._dtype = next(self.parameters()).dtype
        return self._dtype

class BFModel(BFModule):
    """Base model class for brain-feature models.
    
    The model accepts batches of shape (batch_size, n_electrodes, n_samples) where
    and electrode embedding of shape (n_electrodes, d_model).
    n_samples = sample_timebin_size * n_timebins.

    The model's forward pass must return a tuple where:
    - First element is a batch of shape (batch_size, n_timebins, d_output) containing the model's output
    - Remaining elements can be anything

    The model must implement calculate_pretrain_loss(self, x, y) for pretraining.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, electrode_embeddings):
        pass

    def calculate_pretrain_loss(self, electrode_embeddings, batch):
        pass

    def generate_frozen_evaluation_features(self, batch, electrode_embeddings):
        pass

    def calculate_pretrain_test_loss(self, electrode_data_embedding_class, test_dataloader):
        loss = 0
        n_batches = 0
        for batch, (subject_identifier, trial_id) in test_dataloader:
            trial_id, subject_identifier = trial_id[0], subject_identifier[0] # they are all the same in a batch by design
            
            batch = batch.to(self.device, dtype=self.dtype, non_blocking=True)
            electrode_embedded_data = electrode_data_embedding_class.forward(subject_identifier, batch)

            loss += self.calculate_pretrain_loss(electrode_embedded_data)
            n_batches += 1
        return loss / n_batches

from model_transformers import Transformer
class TransformerModel(BFModel):
    def __init__(self, d_model, sample_timebin_size, n_layers_electrode=5, n_layers_time=5, frequency_cutoff_dim=64):
        super().__init__()
        self.d_model = d_model
        self.sample_timebin_size = sample_timebin_size
        self.frequency_cutoff_dim = frequency_cutoff_dim

        self.electrode_transformer = Transformer(d_input=d_model, d_model=d_model, d_output=d_model, n_layer=n_layers_electrode, n_head=12, causal=False, rope=False, cls_token=True)
        self.time_transformer = Transformer(d_input=d_model, d_model=d_model, d_output=d_model, n_layer=n_layers_time, n_head=12, causal=True, rope=True, cls_token=False)
        self.temperature_param = nn.Parameter(torch.tensor(1.0))

    
    def forward(self, embedded_electrode_data, only_electrode_output=False):
        # electrode_embeddings is of shape (n_electrodes, d_model)
        # embedded_electrode_data is of shape (batch_size, n_electrodes, n_timebins, d_model)

        electrode_data = embedded_electrode_data.permute(0, 2, 1, 3) # shape: (batch_size, n_timebins, n_electrodes, d_model)
        batch_size, n_timebins, n_electrodes, d_model = electrode_data.shape

        electrode_output = self.electrode_transformer(electrode_data.reshape(batch_size * n_timebins, n_electrodes, d_model)) # shape: (batch_size*n_timebins, n_electrodes+1, d_model)
        electrode_output = electrode_output[:, 0:1, :].view(batch_size, n_timebins, d_model) # just the CLS token. Shape: (batch_size, n_timebins, d_model)
        if only_electrode_output:
            return electrode_output, None
        
        time_output = self.time_transformer(electrode_output) # shape: (batch_size, n_timebins, d_model)
        return electrode_output, time_output


    def calculate_pretrain_loss(self, electrode_embedded_data, p_electrodes_per_stream=0.5):
        batch_size, n_electrodes, n_timebins, d_model = electrode_embedded_data.shape

        permutation = torch.randperm(n_electrodes)
        electrode_embedded_data = electrode_embedded_data[:, permutation]

        n_electrodes_per_stream = int(n_electrodes * p_electrodes_per_stream)

        # all model outputs shape: (batch_size, n_timebins, d_model)
        _, o1_t = self(electrode_embedded_data[:, :n_electrodes_per_stream, :-1, :])
        o2_e, _ = self(electrode_embedded_data[:, -n_electrodes_per_stream:, 1:, :], only_electrode_output=True)
        similarity = torch.matmul(o1_t[:, :].permute(1, 0, 2), o2_e[:, :].permute(1, 2, 0)) * self.temperature_param

        expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(n_timebins-1, 1).to(self.device, dtype=torch.long).reshape(-1)
        return nn.functional.cross_entropy(similarity.view(-1, batch_size), expanded_arange)
    
    def generate_frozen_evaluation_features(self, electrode_embedded_data):
        batch_size, n_electrodes, n_timebins, d_model = electrode_embedded_data.shape
        return self(electrode_embedded_data, only_electrode_output=True)[0].mean(dim=1)