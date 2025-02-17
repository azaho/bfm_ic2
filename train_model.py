import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from braintreebank_dataset import BrainTreebankSubjectTrialDataset
from braintreebank_subject import BrainTreebankSubject
import wandb, os
import random
import psutil
import time
from muon import Muon
import ml_dtypes

from model_model import LinearModel, BFMModel_Scuffed
from model_electrode_embedding import ElectrodeEmbeddings_LinearModel, ElectrodeEmbeddings_Learned
from model_btbench_evaluation import FrozenModelEvaluation_SS_SM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

all_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]
all_subject_trials = [("btbank" + str(subject_id), trial_id) for subject_id, trial_id in all_subject_trials]
eval_subject_trials = [(1, 2), (2, 6), (3, 0), (6, 4), (7, 0), (4, 1), (10, 0)]
eval_subject_trials = [("btbank" + str(subject_id), trial_id) for subject_id, trial_id in eval_subject_trials]
train_subject_trials = []
for subject_trial in all_subject_trials:
    if subject_trial not in eval_subject_trials:
        train_subject_trials.append(subject_trial)


training_config = {
    'batch_size': 100,

    'n_epochs': 200,
    'learning_rate': 0.0015,
    'weight_decay': 0.0,
    'save_model_every_n_epochs': 1,
    'wandb_project': 'bfm_ic2_0',
    'optimizer': 'Muon',
    'p_test': 0.1,
    # 'train_subject_trials': [("btbank3", 1)],#, ("btbank3", 2)],
    # 'eval_subject_trials': [("btbank3", 0)],
    'train_subject_trials': train_subject_trials,
    'eval_subject_trials': eval_subject_trials,
    'cache_subjects': True, # XXX add random string
}
model_config = {
    'd_model': 192,
    'sample_timebin_size': 256,
    'max_n_timebins': 24,
}
if len(training_config['wandb_project'])==0: wandb = False
n_samples = model_config['max_n_timebins'] * model_config['sample_timebin_size']
dtype = torch.bfloat16

def log(message):
    print(message)

subject_dtype = ml_dtypes.bfloat16 if dtype == torch.bfloat16 else dtype
all_subject_identifiers = [subject_identifier for subject_identifier, trial_id in training_config['train_subject_trials']]
all_subject_identifiers += [subject_identifier for subject_identifier, trial_id in training_config['eval_subject_trials']]
all_subject_identifiers = list(set(all_subject_identifiers))
all_subjects = {}
for subject_identifier in all_subject_identifiers:
    assert "btbank" in subject_identifier, f"Only braintreebank subjects are supported, got {subject_identifier} (need btbankX)"
    subject_id = int(subject_identifier.replace("btbank", ""))
    all_subjects[subject_identifier] = BrainTreebankSubject(subject_id, dtype=subject_dtype, cache=training_config['cache_subjects'], allow_corrupted=False)

log("Loading the datasets...")
datasets = []
for subject_identifier, trial_id in training_config['train_subject_trials']:
    log(f"Loading train dataset for trial {subject_identifier}_{trial_id}...")
    datasets.append(BrainTreebankSubjectTrialDataset(all_subjects[subject_identifier], trial_id, n_samples, dtype=dtype, output_subject_trial_id=True))
    ram_used = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
    log(f"\tTotal RAM used: {ram_used:.2f} GB")
train_datasets = []
test_datasets = []
for dataset in datasets:
    train_size = int(len(dataset) * (1 - training_config['p_test']))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)
train_dataset = ConcatDataset(train_datasets)
test_dataset = ConcatDataset(test_datasets)

# Create dataloaders with custom sampler
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
train_dataloader = DataLoader(
    train_dataset,
    batch_sampler=SubjectBatchSampler(
        [len(ds) for ds in train_datasets],
        batch_size=training_config['batch_size'],
        shuffle=True
    ),
    num_workers=8,  # XXX make a proper varable for this + in eval code
    pin_memory=True,  # Pin memory for faster GPU transfer
    persistent_workers=True  # Keep worker processes alive between iterations
)
test_dataloader = DataLoader(
    test_dataset,
    batch_sampler=SubjectBatchSampler(
        [len(ds) for ds in test_datasets],
        batch_size=training_config['batch_size'],
        shuffle=False
    ),
    num_workers=8,  # XXX make a proper varable for this + in eval code
    pin_memory=True,
    persistent_workers=True
)

log("Done.")
def test_model(model, electrode_embeddings, test_dataloader):
    loss = 0
    n_batches = 0
    for batch_idx, (batch, (subject_identifier, trial_id)) in enumerate(test_dataloader):
        trial_id = trial_id[0]
        subject_identifier = subject_identifier[0] # they are all the same in a batch by design
        
        electrode_embed = electrode_embeddings(subject_identifier)
        batch = batch.to(device, dtype=dtype)
        electrode_embed = electrode_embed.to(device, dtype=dtype)

        loss += model.calculate_loss(electrode_embed, batch)
        n_batches += 1
    return loss / n_batches

# model = LinearModel(model_config['d_model'], model_config['sample_timebin_size']).to(device, dtype=dtype)
# electrode_embeddings = ElectrodeEmbeddings_LinearModel(model_config['d_model'], model_config['sample_timebin_size']).to(device, dtype=dtype)
# electrode_embeddings.add_embedding(subject.subject_identifier, n_electrodes, requires_grad=True)
model = BFMModel_Scuffed(model_config['d_model'], model_config['sample_timebin_size']).to(device, dtype=dtype)
electrode_embeddings = ElectrodeEmbeddings_Learned(model_config['d_model']).to(device, dtype=dtype)
for subject_identifier in all_subject_identifiers:
    electrode_embeddings.add_embedding(subject_identifier, all_subjects[subject_identifier].get_n_electrodes(), requires_grad=True)

eval_subject_trials = [(all_subjects[subject_identifier], trial_id) for subject_identifier, trial_id in training_config['eval_subject_trials']]
evaluation = FrozenModelEvaluation_SS_SM(
    ['onset', 'speech', 'volume', 'pitch'], 
    eval_subject_trials, dtype, 
    training_config['batch_size'] * 4, # for batch size 100, around 4-5 seconds is spent to generate frozen features, and <.8 second to fit regressor and evaluate.
    regression_n_jobs=8,  # XXX make a proper varable for this + in eval code
)

optimizer = torch.optim.Adam(list(model.parameters()) + list(electrode_embeddings.parameters()), lr=training_config['learning_rate'])

all_params = list(model.parameters()) + list(electrode_embeddings.parameters())
optimizers = []
if training_config['optimizer'] == 'Muon':
    matrix_params = [p for p in all_params if p.ndim >= 2]
    other_params = [p for p in all_params if p.ndim < 2]
    optimizers.append(Muon(matrix_params, lr=training_config['learning_rate'], momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5))
    optimizers.append(torch.optim.Adam(other_params, lr=training_config['learning_rate'], weight_decay=training_config['weight_decay']))
else:
    optimizers = [torch.optim.Adam(all_params, lr=training_config['learning_rate'], weight_decay=training_config['weight_decay'])]

if wandb: wandb.init(project=training_config['wandb_project'], name="all_subj5_nocorr", id="all_subj5_nocorr")
for epoch_i in range(training_config['n_epochs']):
    epoch_start_time = time.time()
    epoch_loss = 0
    for batch_idx, (batch, (subject_identifier, trial_id)) in enumerate(train_dataloader):
        model.train()
        for optimizer in optimizers:
            optimizer.zero_grad()
        trial_id = trial_id[0]
        subject_identifier = subject_identifier[0] # they are all the same in a batch by design
        electrode_embed = electrode_embeddings(subject_identifier)

        batch = batch.to(device, dtype=dtype)
        electrode_embed = electrode_embed.to(device, dtype=dtype)

        loss = model.calculate_loss(electrode_embed, batch)
        epoch_loss += loss.item()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        current_time = time.strftime("%H:%M:%S")
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2    # Convert to MB
        process = psutil.Process()
        ram_usage = process.memory_info().rss / 1024**2                 # Convert to MB
        print(f"[{current_time}] Epoch {epoch_i+1}/{training_config['n_epochs']}, Batch {batch_idx+1}/{len(train_dataloader)} ({subject_identifier}_{trial_id}), Loss: {loss.item():.4f}, GPU Memory: {gpu_memory_allocated:.1f}MB:{gpu_memory_reserved:.1f}MB, RAM: {ram_usage:.1f}MB")

    eval_results = {"train_loss": epoch_loss / len(train_dataloader)}
    model.eval()
    with torch.no_grad():
        print("Evaluating on test set...")
        eval_results.update({"test_loss": test_model(model, electrode_embeddings, test_dataloader)})

        print("Evaluating on all metrics...")
        evaluation_results_strings = evaluation.evaluate_on_all_metrics(model, electrode_embeddings, verbose=False, quick_eval=True)
        print(evaluation_results_strings)
        eval_results.update(evaluation_results_strings)
        
        # Calculate and print time remaining
        epoch_time = time.time() - epoch_start_time
        epochs_remaining = training_config['n_epochs'] - (epoch_i + 1)
        time_remaining = epoch_time * epochs_remaining
        hours = int(time_remaining // 3600)
        minutes = int((time_remaining % 3600) // 60)
        seconds = int(time_remaining % 60)
        print(f"Estimated time remaining: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    if wandb: wandb.log(eval_results)

    if (epoch_i+1) % training_config['save_model_every_n_epochs'] == 0:
        model_path = f"models_data/model_epoch_{epoch_i+1}.pth"
        os.makedirs("models_data", exist_ok=True)
        torch.save({
            'eval_results': eval_results,
            'epoch': epoch_i,
            'model_state_dict': model.state_dict(),
            'electrode_embeddings_state_dict': electrode_embeddings.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
if wandb: wandb.finish()