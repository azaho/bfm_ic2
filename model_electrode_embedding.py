import torch
import torch.nn as nn
from model_model import BFModule
    

class ElectrodeEmbeddings(BFModule):
    def __init__(self):
        super(ElectrodeEmbeddings, self).__init__()
        # Every key must be a unique string identifier for a subject
        #   and be a NumPy array of shape (n_electrodes, *) where * is any number of additional dimensions of any size
        self.embeddings = nn.ParameterDict({})
    
    def forward(self, subject_identifier, permutation=None):
        assert subject_identifier in self.embeddings, f"Subject identifier {subject_identifier} not found in embeddings"
        embedding = self.embeddings[subject_identifier]
        if permutation is not None: embedding = embedding[permutation]
        return embedding
    
    def add_embedding(self, subject_identifier, embedding_init, requires_grad=True):
        assert subject_identifier not in self.embeddings, f"Subject identifier {subject_identifier} already in embeddings"
        self.embeddings[subject_identifier] = nn.Parameter(embedding_init, requires_grad=requires_grad)


class ElectrodeEmbeddings_Learned(ElectrodeEmbeddings):
    def __init__(self, d_model, embedding_dim=None, embedding_fanout_requires_grad=True):
        super(ElectrodeEmbeddings_Learned, self).__init__()
        self.embedding_dim = embedding_dim if embedding_dim is not None else d_model
        self.d_model = d_model
        if self.embedding_dim < d_model:
            self.linear_embed = nn.Linear(self.embedding_dim, self.d_model)
            self.linear_embed.weight.requires_grad = embedding_fanout_requires_grad
            self.linear_embed.bias.requires_grad = embedding_fanout_requires_grad
        else: 
            self.linear_embed = lambda x: x # just identity function if embedding dim is already at d_model
    
    def add_embedding(self, subject_identifier, n_electrodes, embedding_init=None, requires_grad=True):
        if embedding_init is None: embedding_init = torch.zeros(n_electrodes, self.embedding_dim)
        super(ElectrodeEmbeddings_Learned, self).add_embedding(subject_identifier, embedding_init, requires_grad)
    
    def forward(self, subject_identifier, permutation=None):
        embedding = super(ElectrodeEmbeddings_Learned, self).forward(subject_identifier, permutation)
        return self.linear_embed(embedding)
    
    
class ElectrodeEmbeddings_Coordinate(ElectrodeEmbeddings_Learned):
    def __init__(self, d_model, embedding_dim=None, embedding_fanout_requires_grad=True):
        super(ElectrodeEmbeddings_Coordinate, self).__init__(d_model, embedding_dim=embedding_dim, 
                                                             embedding_fanout_requires_grad=embedding_fanout_requires_grad)
    
    def add_embedding(self, subject_identifier, n_electrodes, electrode_coordinates, requires_grad=False):
        """
            "Electrode Coordinates" must be LPI coordinates, normalized to [0,1] range given min/max
        """
        assert electrode_coordinates.shape == (n_electrodes, 3), f"Electrode coordinates must be of shape (n_electrodes, 3), got {electrode_coordinates.shape}"

        freq = 200 ** torch.linspace(0, 1, self.embedding_dim//6)

        # Calculate position encodings for each coordinate dimension
        l_enc = electrode_coordinates[:, 0:1] @ freq.unsqueeze(0)  # For L coordinate
        i_enc = electrode_coordinates[:, 1:2] @ freq.unsqueeze(0)  # For I coordinate  
        p_enc = electrode_coordinates[:, 2:3] @ freq.unsqueeze(0)  # For P coordinate
        padding_zeros = torch.zeros(n_electrodes, self.embedding_dim-6*(self.embedding_dim//6)) # padding in case model dimension is not divisible by 6

        # Combine sin and cos encodings
        embedding = torch.cat([torch.sin(l_enc), torch.cos(l_enc), torch.sin(i_enc), torch.cos(i_enc), torch.sin(p_enc), torch.cos(p_enc), padding_zeros], dim=-1)
        super(ElectrodeEmbeddings_Coordinate, self).add_embedding(subject_identifier, n_electrodes, embedding_init=embedding, requires_grad=requires_grad)


class ElectrodeEmbeddings_LinearModel(ElectrodeEmbeddings):
    def __init__(self, d_model, sample_timebin_size):
        super(ElectrodeEmbeddings_LinearModel, self).__init__()
        self.d_model = d_model
        self.sample_timebin_size = sample_timebin_size

    def add_embedding(self, subject_identifier, n_electrodes, embedding_init=None, requires_grad=True):
        if embedding_init is None:
            # Using Kaiming initialization, accounting for the total fan-in
            # fan_in = n_electrodes * sample_timebin_size (total input features after reshaping)
            fan_in = n_electrodes * self.sample_timebin_size
            std = (2.0 / fan_in) ** 0.5
            embedding_init = torch.randn(n_electrodes, self.sample_timebin_size, self.d_model) * std # nonzero init to break symmetry
        super(ElectrodeEmbeddings_LinearModel, self).add_embedding(subject_identifier, embedding_init, requires_grad=requires_grad)
        