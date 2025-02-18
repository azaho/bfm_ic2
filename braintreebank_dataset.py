import torch
from torch.utils.data import Dataset
from braintreebank_subject import BrainTreebankSubject
import numpy as np
import ml_dtypes

class BrainTreebankSubjectTrialDataset(Dataset):
    def __init__(self, subject, trial_id, window_size, dtype=torch.float32, output_subject_trial_id=False):
        """
        Args:
            subject (BrainTreebankSubject): Subject object
            trial_id (int): Trial ID
            dtype (torch.dtype): Data type to load the data in (float32, bfloat16)
            window_size (int): Number of time samples per data item
        """
        self.subject = subject
        self.trial_id = trial_id
        self.window_size = window_size
        self.dtype = dtype
        self.output_subject_trial_id = output_subject_trial_id

        subject.load_neural_data(trial_id)
        self.n_windows = self.subject.electrode_data_length[trial_id] // self.window_size
    
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, idx):
        start_idx = idx * self.window_size
        end_idx = start_idx + self.window_size
        window = self.subject.get_all_electrode_data(self.trial_id, start_idx, end_idx)
        if window.dtype == ml_dtypes.bfloat16:
            window = window.astype(np.float32) # XXX this is a hack to make ml_dtypes work with torch tensor conversion
        if self.output_subject_trial_id:
            return torch.from_numpy(window).to(dtype=self.dtype), (self.subject.subject_identifier, self.trial_id)
        else:
            return torch.from_numpy(window).to(dtype=self.dtype)


if __name__ == "__main__":
    subject = BrainTreebankSubject(3, cache=False)
    dataset = BrainTreebankSubjectTrialDataset(subject, 0, 100, torch.float32)
    print(len(dataset))
    print(dataset[0].shape)
