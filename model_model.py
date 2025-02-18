import torch
import torch.nn as nn
from model_utils import BFModule

# the Model class needs to accept a batch of shape (batch_size, n_electrodes, n_samples)
#    where n_samples = sample_timebin_size * n_timebins
# Model needs to output a TUPLE where the first element is a batch 
#   of shape (batch_size, n_timebins, d_output) - the model's output
#   the rest can be anything
# 
# Model needs to contain a method calculate_pretrain_loss(self, x, y) 

class LinearModel(BFModule):
    def __init__(self, d_model, sample_timebin_size):
        super(LinearModel, self).__init__()
        self.sample_timebin_size = sample_timebin_size
        self.linear_dynamics = nn.Linear(d_model, d_model)

    def forward(self, x, electrode_embeddings):
        # electrode_embeddings is a batch of shape (n_electrodes, sample_timebin_size, d_model)
        batch_size, n_electrodes, n_samples = x.size()
        n_timebins = n_samples // self.sample_timebin_size
        
        x = x.reshape(batch_size, n_electrodes, n_timebins, self.sample_timebin_size)
        x = x.permute(0, 2, 1, 3).contiguous()  # shape: (batch_size, n_timebins, n_electrodes, sample_timebin_size)
        x = x.reshape(batch_size, n_timebins, -1)  # shape: (batch_size, n_timebins, n_electrodes * sample_timebin_size)

        electrode_embeddings = electrode_embeddings.reshape(-1, electrode_embeddings.size(-1))  # shape: (n_electrodes * sample_timebin_size, d_model)
        x = torch.matmul(x, electrode_embeddings)  # shape: (batch_size, n_timebins, d_model)

        return x, self.linear_dynamics(x)
        
    def calculate_loss(self, electrode_embed, batch):
        # electrode_embeddings is a batch of shape (n_electrodes, sample_timebin_size, d_model)
        n_electrodes, sample_timebin_size, d_model = electrode_embed.shape
        batch_size, n_electrodes, n_samples = batch.shape
        n_timebins = n_samples // sample_timebin_size

        permutation = torch.randperm(n_electrodes)
        batch = batch[:, permutation]
        electrode_embed = electrode_embed[permutation]

        # all model outputs shape: (batch_size, n_timebins, d_model)
        o1 = self(batch[:, :n_electrodes//2, :-sample_timebin_size], electrode_embed[:n_electrodes//2])[1]
        o2 = self(batch[:, n_electrodes//2:, sample_timebin_size:], electrode_embed[n_electrodes//2:])[0]
        similarity = torch.matmul(o1[:, :].permute(1, 0, 2), o2[:, :].permute(1, 2, 0)) / 10000

        expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(n_timebins-1, 1).to(self.device, dtype=torch.long).reshape(-1)
        return nn.functional.cross_entropy(similarity.view(-1, batch_size), expanded_arange)
    
    def generate_frozen_evaluation_features(self, batch, electrode_embed):
        # generate frozen features
        frozen_features = self(batch[:, :, :], electrode_embed[:])[0].mean(dim=1)
        return frozen_features
    

from training_architecture_juice import ElectrodeTransformer, TimeTransformer
class BFMModel_Scuffed(BFModule):
    def __init__(self, d_model, sample_timebin_size):
        super().__init__()
        self.d_model = d_model
        self.sample_timebin_size = sample_timebin_size

        transformer_config = {
            'max_n_electrodes': 128,
            'n_freq_features': 64,
            'max_n_time_bins': 12,
            'd_model': d_model,
            'n_heads': 12,
            'n_layers': 10,
            'dropout': 0.2,
            'dim_output': d_model,
            'dim_input': 64,
            'dtype': torch.bfloat16,  # Changed dtype to bfloat16 to match error
            'electrode_embedding': 'normal', # options: 'normal', 'zeros', 'coordinates'
            'electrode_embedding_grad': True,
        }
        transformer_config['n_layers_electrode'] = transformer_config['n_layers']//2
        transformer_config['n_layers_time'] = transformer_config['n_layers'] - transformer_config['n_layers_electrode']
        transformer_config['rope_encoding_scale'] = transformer_config['max_n_time_bins']
        self.electrode_transformer = ElectrodeTransformer(config=transformer_config)
        self.time_transformer = TimeTransformer(config=transformer_config)
        self.temperature_param = nn.Parameter(torch.tensor(1.0))
        self.transformer_config = transformer_config



    def forward(self, x, electrode_embed):
        # electrode_embeddings is of shape (n_electrodes, d_model)
        # x is of shape (batch_size, n_electrodes, n_samples)
        #   where n_samples = sample_timebin_size * n_timebins
        batch_size, n_electrodes, n_samples = x.shape
        n_timebins = n_samples // self.sample_timebin_size

        # Reshape x to separate timebins and samples within each timebin
        x = x.reshape(batch_size, n_electrodes, n_timebins, self.sample_timebin_size)
        
        # Calculate FFT for each timebin
        x = x.reshape(-1, self.sample_timebin_size)
        x = torch.fft.rfft(x, dim=-1)  # Using rfft for real-valued input
        x = x.reshape(batch_size, n_electrodes, n_timebins, -1)  # shape: (batch_size, n_electrodes, n_timebins, n_freq)
        x = x[:, :, :, :self.transformer_config['dim_input']]
        # Calculate magnitude (equivalent to scipy.signal.stft's magnitude)
        x = torch.abs(x).to(dtype=self.dtype) # XXX: anoying that need to do this, convert it back to bfloat16, fft outputs float32
        # Convert to power in dB
        #x = 20 * torch.log10(x.pow(2) + 1e-10)  # adding small constant to prevent log(0)
        # do scuffed log (like in my previous code)
        x = 10 * torch.log(x + 1e-10)

        # Rearrange dimensions to match expected shap
        x = x.permute(0, 2, 1, 3)  # shape: (batch_size, n_timebins, n_electrodes, n_freq)
        x = x.unsqueeze(1) # add n_samples dimension

        electrode_output = self.electrode_transformer(x, electrode_embed)
        electrode_output = electrode_output[:, :, :, 0:1, :] # just the CLS token
        electrode_output = electrode_output.transpose(2, 3) #XXX scuffed, probably bug, but keeping like in the old code.
        time_output = self.time_transformer(electrode_output)

        electrode_output = electrode_output.view(batch_size, n_timebins, self.transformer_config['dim_output']) 
        time_output = time_output.view(batch_size, n_timebins, self.transformer_config['dim_output']) 
        return time_output, electrode_output

    def calculate_loss(self, electrode_embed, batch):
        # electrode_embeddings is a batch of shape (n_electrodes, sample_timebin_size, d_model)
        batch_size, n_electrodes, n_samples = batch.shape
        n_timebins = n_samples // self.sample_timebin_size

        permutation = torch.randperm(n_electrodes)
        batch = batch[:, permutation]
        electrode_embed = electrode_embed[permutation]

        # all model outputs shape: (batch_size, n_timebins, d_model)
        o1_t, o1_e = self(batch[:, :n_electrodes//2, :-self.sample_timebin_size], electrode_embed[:n_electrodes//2])
        o2_t, o2_e = self(batch[:, n_electrodes//2:, self.sample_timebin_size:], electrode_embed[n_electrodes//2:])
        similarity = torch.matmul(o1_t[:, :].permute(1, 0, 2), o2_e[:, :].permute(1, 2, 0)) * self.temperature_param

        expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(n_timebins-1, 1).to(self.device, dtype=torch.long).reshape(-1)
        return nn.functional.cross_entropy(similarity.view(-1, batch_size), expanded_arange)
    
    def generate_frozen_evaluation_features(self, batch, electrode_embed):
        # generate frozen features
        frozen_features = self(batch[:, :, :], electrode_embed[:])[0].mean(dim=1)
        return frozen_features

