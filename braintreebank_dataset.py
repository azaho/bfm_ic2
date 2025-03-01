import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from braintreebank_subject import BrainTreebankSubject
from train_utils import log
from multiprocessing import Pool
import torch.multiprocessing as mp
import random

# Standardizing pretraining and evaluation subjects and trials
all_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]
all_subject_trials = [("btbank" + str(subject_id), trial_id) for subject_id, trial_id in all_subject_trials]
eval_subject_trials = [(1, 2), (2, 6), (3, 0), (6, 4), (7, 0), (4, 1), (10, 0)]
eval_subject_trials = [("btbank" + str(subject_id), trial_id) for subject_id, trial_id in eval_subject_trials]
train_subject_trials = [st for st in all_subject_trials if st not in eval_subject_trials]

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
        window = self.subject.get_all_electrode_data(self.trial_id, start_idx, end_idx).to(dtype=self.dtype)
        if self.output_subject_trial_id: 
            return window, (self.subject.subject_identifier, self.trial_id)
        else: return window


def _load_single_dataset(args):
    all_subjects, subject_identifier, trial_id, n_samples, dtype = args
    try:
        log(f"loading dataset for {subject_identifier}_{trial_id}...", indent=1, priority=1)
        dataset = BrainTreebankSubjectTrialDataset(
            all_subjects[subject_identifier], 
            trial_id, 
            n_samples, 
            dtype=dtype, 
            output_subject_trial_id=True
        )
        log(f"finished loading dataset for {subject_identifier}_{trial_id}", indent=1, priority=1)
        return dataset
    except Exception as e:
        log(f"Error loading dataset for {subject_identifier}_{trial_id}: {str(e)}", indent=1, priority=1)
        raise e

def load_dataloaders(train_subject_trials, eval_subject_trials, p_test, n_samples, dtype, batch_size, num_workers_init=14, num_workers_dataloaders=12, prefetch_factor=2, cache=True, allow_corrupted=False, test_num_workers_fraction=0.1):
    # Step 1: Load all subjects
    all_subject_identifiers = [subject_identifier for subject_identifier, trial_id in train_subject_trials]
    all_subject_identifiers += [subject_identifier for subject_identifier, trial_id in eval_subject_trials]
    all_subject_identifiers = list(set(all_subject_identifiers))
    all_subjects = {}
    for subject_identifier in all_subject_identifiers:
        assert "btbank" in subject_identifier, f"Only braintreebank subjects are supported, got {subject_identifier} (need btbankX)"
        subject_id = int(subject_identifier.replace("btbank", ""))
        all_subjects[subject_identifier] = BrainTreebankSubject(subject_id, dtype=dtype, cache=cache, allow_corrupted=allow_corrupted)

    # Step 2: Load all datasets in parallel (if cache is enabled)
    pool_params = [(all_subjects, subject_identifier, trial_id, n_samples, dtype) for subject_identifier, trial_id in train_subject_trials]
    if cache and num_workers_init > 1:
        n_processes = min(num_workers_init, len(train_subject_trials))
        log(f"Loading {len(train_subject_trials)} datasets in parallel... with {n_processes} processes")
        datasets = []
        mp.set_sharing_strategy('file_descriptor')
        with Pool(processes=n_processes, maxtasksperchild=1) as pool:
            try:
                # Use imap instead of map_async for better progress tracking
                for dataset in pool.imap(_load_single_dataset, pool_params):
                    datasets.append(dataset)
                log("Done loading all datasets.")
            except Exception as e:
                log(f"Error in parallel loading: {str(e)}")
                pool.terminate()
                raise e
    else:
        datasets = [_load_single_dataset(args) for args in pool_params]

    # Step 3: Split into train and test
    train_datasets = []
    test_datasets = []
    for dataset in datasets:
        train_size = int(len(dataset) * (1 - p_test))
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)

    # Step 4: Create dataloaders with custom sampler
    class SubjectBatchSampler(torch.utils.data.Sampler):
        def __init__(self, dataset_sizes, batch_size, shuffle=True):
            self.dataset_sizes = dataset_sizes
            self.batch_size = batch_size
            self.shuffle = shuffle

        def __iter__(self):
            # Create batches for each subject
            all_batches = []
            start_idx = 0
            
            for size in self.dataset_sizes:
                # Create indices for this subject
                subject_indices = list(range(start_idx, start_idx + size))
                if self.shuffle:
                    random.shuffle(subject_indices)
                
                # Create batches
                subject_batches = [subject_indices[i:i + self.batch_size] 
                                for i in range(0, len(subject_indices), self.batch_size)]
                all_batches.extend(subject_batches)
                start_idx += size
            
            # Shuffle the order of batches if needed
            if self.shuffle:
                random.shuffle(all_batches)
                
            return iter(all_batches)
        
        def __len__(self):
            return sum((size + self.batch_size - 1) // self.batch_size 
                    for size in self.dataset_sizes)
    num_workers_dataloader_test = max(int(num_workers_dataloaders * test_num_workers_fraction), 1)
    num_workers_dataloader_train = num_workers_dataloaders - num_workers_dataloader_test
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=SubjectBatchSampler(
            [len(ds) for ds in train_datasets],
            batch_size=batch_size,
            shuffle=True
        ),
        num_workers=num_workers_dataloader_train,
        pin_memory=True,  # Pin memory for faster GPU transfer
        persistent_workers=True,  # Keep worker processes alive between iterations
        prefetch_factor=prefetch_factor
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=SubjectBatchSampler(
            [len(ds) for ds in test_datasets],
            batch_size=batch_size,
            shuffle=False
        ),
        num_workers=num_workers_dataloader_test,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    return all_subjects, train_dataloader, test_dataloader


if __name__ == "__main__":
    subject = BrainTreebankSubject(3, cache=False)
    dataset = BrainTreebankSubjectTrialDataset(subject, 0, 100, torch.float32)
    print(len(dataset))
    print(dataset[0].shape)
