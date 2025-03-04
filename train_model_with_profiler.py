import torch
import wandb, os
import time
import torch.profiler

from model_model import TransformerModel
from model_electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit, ElectrodeDataEmbeddingFFT
from braintreebank_dataset import load_dataloaders

from model_btbench_evaluation import FrozenModelEvaluation_SS_SM
from braintreebank_dataset import train_subject_trials, eval_subject_trials

from muon import Muon
from train_utils import log, update_dir_name, update_random_seed, parse_configs_from_args

training_config = {
    'n_epochs': 1,
    'p_test': 0.1,

    'optimizer': 'Muon',
    'batch_size': 100,
    'learning_rate': 0.002,
    'weight_decay': 0.0,
    'p_electrodes_per_stream': 0.5,
    
    # 'train_subject_trials': [("btbank3", 1), ("btbank3", 2)],
    # 'eval_subject_trials': [("btbank3", 0)],
    # 'train_subject_trials': train_subject_trials,
    # 'eval_subject_trials': eval_subject_trials,

    # MINI-BFM
    'train_subject_trials': [("btbank1", 0), ("btbank1", 1), ("btbank2", 4), ("btbank2", 5), ("btbank3", 1), ("btbank3", 2), ("btbank7", 1), ("btbank10", 1)],
    'eval_subject_trials': [("btbank1", 2), ("btbank2", 6), ("btbank3", 0), ("btbank7", 0), ("btbank10", 0)],
    
    'data_dtype': torch.float16,

    'random_string': "TEMP",
}
model_config = {
    'sample_timebin_size': 256,
    'max_frequency_bin': 64,
    'max_n_timebins': 24,
    'max_n_electrodes': 128,

    'init_normalization': True, # XXX rename to a more sensible name later

    'electrode_embedding': {
        'type': 'learned', # coordinate_init, noisy_coordinate, learned
        'coordinate_noise_std': 0.0, # only relevant for noisy_coordinate type; note coordinates are normalized to be within [0,1]
        'embedding_dim': None,
    },

    'dtype': torch.bfloat16,

    'transformer': {
        'd_model': 192,
        'n_heads': 12,
        'n_layers_electrode': 5,
        'n_layers_time': 5,
        'dropout': 0.2,
    },
}
cluster_config = {
    'save_model_every_n_epochs': 20,
    'eval_model_every_n_epochs': 1,

    'wandb_project': '',
    'timestamp': time.strftime("%Y%m%d_%H%M%S"),

    'cache_subjects': False,

    'num_workers_init': 1,
    'num_workers_dataloaders': 30,
    'num_workers_eval': 4,
    'prefetch_factor': 2,
}
parse_configs_from_args(training_config, model_config, cluster_config)
if len(cluster_config['wandb_project'])==0: wandb = False
dir_name = update_dir_name(model_config, training_config, cluster_config)
update_random_seed(training_config)
cluster_config['wandb_name'] = cluster_config['dir_name']
log(f"Directory name: {dir_name}", priority=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Using device: {device}", priority=0)


log(f"Loading dataloaders...", priority=0)
n_samples = model_config['max_n_timebins'] * model_config['sample_timebin_size']
all_subjects, train_dataloader, test_dataloader = load_dataloaders(
    training_config['train_subject_trials'], training_config['eval_subject_trials'], training_config['p_test'], n_samples, training_config['data_dtype'], training_config['batch_size'],
    num_workers_init=cluster_config['num_workers_init'], num_workers_dataloaders=cluster_config['num_workers_dataloaders'], 
    cache=cluster_config['cache_subjects'], allow_corrupted=False,
    prefetch_factor=cluster_config['prefetch_factor'],
)


model = TransformerModel(
    model_config['transformer']['d_model'], 
    model_config['sample_timebin_size'], 
    n_layers_electrode=model_config['transformer']['n_layers_electrode'], 
    n_layers_time=model_config['transformer']['n_layers_time']).to(device, dtype=model_config['dtype'])

if model_config['electrode_embedding']['type'] == 'learned':
    electrode_embeddings = ElectrodeEmbedding_Learned(
        model_config['transformer']['d_model'], 
        embedding_dim=model_config['electrode_embedding']['embedding_dim']
    )
elif model_config['electrode_embedding']['type'] == 'coordinate_init':
    electrode_embeddings = ElectrodeEmbedding_Learned_CoordinateInit(
        model_config['transformer']['d_model'], 
        embedding_dim=model_config['electrode_embedding']['embedding_dim']
    )
elif model_config['electrode_embedding']['type'] == 'noisy_coordinate':
    electrode_embeddings = ElectrodeEmbedding_NoisyCoordinate(
        model_config['transformer']['d_model'], 
        coordinate_noise_std=model_config['electrode_embedding']['coordinate_noise_std'],
        embedding_dim=model_config['electrode_embedding']['embedding_dim']
    )
else:
    raise ValueError(f"Invalid electrode embedding type: {model_config['electrode_embedding']['type']}")
electrode_embeddings = electrode_embeddings.to(device, dtype=model_config['dtype'])

electrode_data_embeddings = ElectrodeDataEmbeddingFFT(
    electrode_embeddings, model_config['sample_timebin_size'], 
    max_frequency_bin=model_config['max_frequency_bin'], max_n_electrodes=model_config['max_n_electrodes']
).to(device, dtype=model_config['dtype'])
for subject in all_subjects.values():
    log(f"Adding subject {subject.subject_identifier} to electrode data embeddings...", priority=0)
    trial_id = next(trial_id for (sub_id, trial_id) in training_config['train_subject_trials'] if sub_id == subject.subject_identifier)
    electrode_data_embeddings.add_subject(subject, init_normalization=model_config['init_normalization'], 
                                          init_normalization_window_to=2048 * 60 * 5, init_normalization_trial_id=trial_id)
electrode_data_embeddings = electrode_data_embeddings.to(device, dtype=model_config['dtype']) # moving to device again to ensure the new parameters are on the correct device

eval_subject_trials = [(all_subjects[subject_identifier], trial_id) for subject_identifier, trial_id in training_config['eval_subject_trials']]
evaluation = FrozenModelEvaluation_SS_SM(
    ['speech', 'volume'], eval_subject_trials, 
    training_config['data_dtype'], training_config['batch_size'] * 2, # Can have a bigger batch size here if that speeds things up
    num_workers_eval=cluster_config['num_workers_eval'],
    prefetch_factor=cluster_config['prefetch_factor'],
)


all_params = list(model.parameters()) + list(electrode_data_embeddings.parameters())
n_model_params = sum(p.numel() for p in model.parameters())
n_embed_params = sum(p.numel() for p in electrode_data_embeddings.parameters())
log(f"Model parameters: {n_model_params:,}", priority=0)
log(f"Embedding parameters: {n_embed_params:,}", priority=0)
log(f"Total parameters: {n_model_params + n_embed_params:,}", priority=0)
model_config['n_params'] = {
    'model': n_model_params,
    'embeddings': n_embed_params,
    'total': n_model_params + n_embed_params
}

optimizers = []
if training_config['optimizer'] == 'Muon':
    matrix_params = [p for p in all_params if p.ndim >= 2]
    other_params = [p for p in all_params if p.ndim < 2]

    optimizers.append(Muon(matrix_params, lr=training_config['learning_rate'], momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5, weight_decay=training_config['weight_decay']))
    optimizers.append(torch.optim.AdamW(other_params, lr=training_config['learning_rate'], weight_decay=training_config['weight_decay']))
else:
    optimizers = [torch.optim.AdamW(all_params, lr=training_config['learning_rate'], weight_decay=training_config['weight_decay'])]



if wandb: 
    wandb.init(project=cluster_config['wandb_project'], name=cluster_config['wandb_name'], id=cluster_config['wandb_name'],
    config={"training_config": training_config, "model_config": model_config, "cluster_config": cluster_config}, settings=wandb.Settings(init_timeout=480))

# Set up PyTorch profiler
use_profiler = True  # Set to False to disable profiling
profile_batches = 100  # Number of batches to profile
profiler = None

if use_profiler:
    log(f"Setting up PyTorch profiler to profile {profile_batches} batches...", priority=0)
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=1,
            active=profile_batches,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./profiler_logs/{cluster_config['dir_name']}"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,  # Track module hierarchy
        with_flops=True     # Estimate FLOPs where possible
    )
    # Start profiler before data loading to capture data movement operations
    profiler.start()

# Wrap the entire epoch in profiling context
for epoch_i in range(training_config['n_epochs']):
    epoch_start_time = time.time()
    model.train()

    # Main training loop
    epoch_loss = 0
    for batch_idx, (batch, (subject_identifier, trial_id)) in enumerate(train_dataloader):
        # Add explicit markers for data loading and preparation
        if profiler is not None:
            profiler.step()
            torch.cuda.synchronize()  # Ensure CUDA operations are complete
        
        for optimizer in optimizers: optimizer.zero_grad()
        subject_identifier, trial_id = subject_identifier[0], trial_id[0] # they are all the same in a batch by design
        
        # Add marker for data transfer to device
        batch_transfer_start = time.time()
        batch = batch.to(device, dtype=model_config['dtype'], non_blocking=True)
        if profiler is not None and batch_idx < profile_batches + 1:
            log(f"Data to device time: {time.time() - batch_transfer_start:.4f}s", priority=0)
        
        # Forward pass
        forward_start = time.time()
        electrode_embedded_data = electrode_data_embeddings.forward(subject_identifier, batch)
        loss = model.calculate_pretrain_loss(electrode_embedded_data, p_electrodes_per_stream=training_config['p_electrodes_per_stream'])
        if profiler is not None and batch_idx < profile_batches + 1:
            log(f"Forward pass time: {time.time() - forward_start:.4f}s", priority=0)
        
        epoch_loss += loss.item()

        # Backward pass
        backward_start = time.time()
        loss.backward()
        if profiler is not None and batch_idx < profile_batches + 1:
            log(f"Backward pass time: {time.time() - backward_start:.4f}s", priority=0)
        
        # Optimizer step
        optim_start = time.time()
        for optimizer in optimizers: optimizer.step()
        if profiler is not None and batch_idx < profile_batches + 1:
            log(f"Optimizer step time: {time.time() - optim_start:.4f}s", priority=0)

        log(f"Epoch {epoch_i+1}/{training_config['n_epochs']}, Batch {batch_idx+1}/{len(train_dataloader)} ({subject_identifier}_{trial_id}), Loss: {loss.item():.4f}", priority=0)
        
        # Step the profiler
        if use_profiler and profiler is not None:
            # We already called step() at the beginning of the loop
            # Stop profiling after specified number of batches
            if batch_idx >= profile_batches:
                if profiler is not None:
                    profiler.stop()
                    log(f"Profiling completed. Results saved to ./profiler_logs/{cluster_config['dir_name']}", priority=0)
                    log("To view the flame graph, run: tensorboard --logdir=./profiler_logs", priority=0)
                    profiler = None
                    use_profiler = False
    
    epoch_loss /= len(train_dataloader)

    # Evaluate the model
    model.eval()
    eval_results = {"train_loss": epoch_loss}
    with torch.no_grad():
        eval_results.update({"test_loss": model.calculate_pretrain_test_loss(electrode_data_embeddings, test_dataloader)})
        if (epoch_i+1) % cluster_config['eval_model_every_n_epochs'] == 0:
            evaluation_results_strings = evaluation.evaluate_on_all_metrics(model, electrode_data_embeddings, log_priority=1, quick_eval=True)
            eval_results.update(evaluation_results_strings)
        time_remaining = (time.time() - epoch_start_time) * (training_config['n_epochs'] - (epoch_i + 1))
        log(f"Estimated time remaining: {time.strftime('%H:%M:%S', time.gmtime(time_remaining))}", priority=0)
    if wandb: wandb.log(eval_results)

    # Save the model
    if (epoch_i+1) % cluster_config['save_model_every_n_epochs'] == 0:
        model_path = f"models_data/{cluster_config['dir_name']}/model_epoch_{epoch_i+1}.pth"
        os.makedirs(f"models_data/{cluster_config['dir_name']}", exist_ok=True)
        # Convert torch dtypes to strings before saving
        def convert_dtypes(config):
            if isinstance(config, dict):
                return {k: convert_dtypes(v) for k, v in config.items()}
            elif isinstance(config, torch.dtype):
                return str(config)
            return config
            
        torch.save({
            'eval_results': eval_results,
            'epoch': epoch_i,
            'model_state_dict': model.state_dict(),
            'electrode_data_embeddings_state_dict': electrode_data_embeddings.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_config': convert_dtypes(training_config),
            'model_config': convert_dtypes(model_config), 
            'cluster_config': convert_dtypes(cluster_config),
        }, model_path)

# Make sure profiler is stopped at the end
if use_profiler and profiler is not None:
    profiler.stop()
    log(f"Profiling completed. Results saved to ./profiler_logs/{cluster_config['dir_name']}", priority=0)
    log("To view the flame graph, run: tensorboard --logdir=./profiler_logs", priority=0)

if wandb: wandb.finish()