import torch
import torch.nn as nn


class BFModule(nn.Module):
    """ 
        This module is a base class for all modules that need to be compatible with BFM.
        It ensures that the module stores its device and dtype
    """
    def __init__(self):
        super(BFModule, self).__init__()
        self._device = None
        self._dtype = None
    
    def to(self, *args, **kwargs):
        # Call the parent's to() method first
        output = super().to(*args, **kwargs)
        
        # Parse device and dtype from the arguments
        device, dtype = None, None
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device = torch.device(arg)
            elif isinstance(arg, torch.dtype):
                dtype = arg
        
        if 'device' in kwargs:
            device = torch.device(kwargs['device'])
        if 'dtype' in kwargs:
            dtype = kwargs['dtype']
            
        # Store the values
        if device is not None:
            self._device = device
        if dtype is not None:
            self._dtype = dtype
            
        return output
    
    @property
    def device(self):
        if self._device is None:
            # Get device from first parameter if available
            if len(self.embeddings) > 0:
                first_param = next(iter(self.embeddings.values()))
                self._device = first_param.device
        return self._device
    
    @property
    def dtype(self):
        if self._dtype is None:
            # Get dtype from first parameter if available
            if len(self.embeddings) > 0:
                first_param = next(iter(self.embeddings.values()))
                self._dtype = first_param.dtype
        return self._dtype