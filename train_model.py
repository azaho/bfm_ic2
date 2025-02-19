import torch
import wandb, os
import time

from model_model import LinearModel, BFMModel_Scuffed
from model_electrode_embedding import ElectrodeEmbeddings_LinearModel, ElectrodeEmbeddings_Learned
from braintreebank_dataset import load_dataloaders

from model_btbench_evaluation import FrozenModelEvaluation_SS_SM
from btbench_config import train_subject_trials, eval_subject_trials

from muon import Muon
from utils import log, update_dir_name, update_random_seed

training_config = {
    'n_epochs': 200,
    'p_test': 0.1,

    'optimizer': 'Muon',
    'batch_size': 100,
    'learning_rate': 0.0015,
    'weight_decay': 0.0,
    
    # 'train_subject_trials': [("btbank3", 1), ("btbank3", 2)],
    # 'eval_subject_trials': [("btbank3", 0)],
    'train_subject_trials': train_subject_trials,
    'eval_subject_trials': eval_subject_trials,
    
    'data_dtype': torch.float32,

    'random_string': "X_normfreq_nosqrt",
}
model_config = {
    'sample_timebin_size': 256,
    'max_n_timebins': 24,

    'dtype': torch.bfloat16,

    'transformer': {
        'd_model': 192,
        'embedding_dim': None,
        'n_heads': 12,
        'n_layers_electrode': 5,
        'n_layers_time': 5,
        'dropout': 0.2,
    },
}
cluster_config = {
    'save_model_every_n_epochs': 1,
    'eval_model_every_n_epochs': 2,

    'wandb_project': 'all_subjects_exp',
    'timestamp': time.strftime("%Y%m%d_%H%M%S"),

    'cache_subjects': True, 

    'num_workers_init': 15,
    'num_workers_dataloaders': 12,
    'num_workers_eval': 3,
}
if len(cluster_config['wandb_project'])==0: wandb = False
update_dir_name(model_config, training_config, cluster_config)
update_random_seed(training_config)
cluster_config['wandb_name'] = cluster_config['dir_name']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Using device: {device}", priority=0)


n_samples = model_config['max_n_timebins'] * model_config['sample_timebin_size']
all_subjects, train_dataloader, test_dataloader = load_dataloaders(
    training_config['train_subject_trials'], training_config['eval_subject_trials'], training_config['p_test'], n_samples, training_config['data_dtype'], training_config['batch_size'],
    num_workers_init=cluster_config['num_workers_init'], num_workers_dataloaders=cluster_config['num_workers_dataloaders'], 
    cache=cluster_config['cache_subjects'], allow_corrupted=False,
)


# model = LinearModel(model_config['d_model'], model_config['sample_timebin_size']).to(device, dtype=training_config['dtype'])
# electrode_embeddings = ElectrodeEmbeddings_LinearModel(model_config['d_model'], model_config['sample_timebin_size']).to(device, dtype=training_config['dtype'])
model = BFMModel_Scuffed(model_config['transformer']['d_model'], model_config['sample_timebin_size']).to(device, dtype=model_config['dtype'])
electrode_embeddings = ElectrodeEmbeddings_Learned(model_config['transformer']['d_model'], embedding_dim=model_config['transformer']['embedding_dim']).to(device, dtype=model_config['dtype'])
for subject_identifier in all_subjects.keys():
    electrode_embeddings.add_embedding(subject_identifier, all_subjects[subject_identifier].get_n_electrodes(), requires_grad=True)


eval_subject_trials = [(all_subjects[subject_identifier], trial_id) for subject_identifier, trial_id in training_config['eval_subject_trials']]
evaluation = FrozenModelEvaluation_SS_SM(
    ['speech', 'volume'], eval_subject_trials, 
    training_config['data_dtype'], training_config['batch_size'] * 8,
    regression_n_jobs=cluster_config['num_workers_eval'],
)


all_params = list(model.parameters()) + list(electrode_embeddings.parameters())
optimizers = []
if training_config['optimizer'] == 'Muon':
    matrix_params = [p for p in all_params if p.ndim >= 2]
    other_params = [p for p in all_params if p.ndim < 2]
    optimizers.append(Muon(matrix_params, lr=training_config['learning_rate'], momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5, weight_decay=training_config['weight_decay']))
    optimizers.append(torch.optim.AdamW(other_params, lr=training_config['learning_rate'], weight_decay=training_config['weight_decay']))
else:
    optimizers = [torch.optim.AdamW(all_params, lr=training_config['learning_rate'], weight_decay=training_config['weight_decay'])]




if wandb: wandb.init(project=cluster_config['wandb_project'], name=cluster_config['wandb_name'], id=cluster_config['wandb_name'])
for epoch_i in range(training_config['n_epochs']):
    epoch_start_time = time.time()
    model.train()

    # Main training loop
    epoch_loss = 0
    for batch_idx, (batch, (subject_identifier, trial_id)) in enumerate(train_dataloader):
        for optimizer in optimizers: optimizer.zero_grad()
        subject_identifier, trial_id = subject_identifier[0], trial_id[0] # they are all the same in a batch by design
        
        electrode_embed = electrode_embeddings(subject_identifier)
        batch = batch.to(device, dtype=model_config['dtype'], non_blocking=True)
        electrode_embed = electrode_embed.to(device, dtype=model_config['dtype'], non_blocking=True)

        loss = model.calculate_pretrain_loss(electrode_embed, batch)
        epoch_loss += loss.item()

        loss.backward()
        for optimizer in optimizers: optimizer.step()

        log(f"Epoch {epoch_i+1}/{training_config['n_epochs']}, Batch {batch_idx+1}/{len(train_dataloader)} ({subject_identifier}_{trial_id}), Loss: {loss.item():.4f}", priority=0)
    epoch_loss /= len(train_dataloader)

    # Evaluate the model
    model.eval()
    eval_results = {"train_loss": epoch_loss}
    with torch.no_grad():
        eval_results.update({"test_loss": model.calculate_pretrain_test_loss(electrode_embeddings, test_dataloader)})
        if (epoch_i+1) % cluster_config['eval_model_every_n_epochs'] == 0:
            evaluation_results_strings = evaluation.evaluate_on_all_metrics(model, electrode_embeddings, log_priority=1, quick_eval=True)
            eval_results.update(evaluation_results_strings)
        time_remaining = (time.time() - epoch_start_time) * (training_config['n_epochs'] - (epoch_i + 1))
        log(f"Estimated time remaining: {time.strftime('%H:%M:%S', time.gmtime(time_remaining))}", priority=0)
    if wandb: wandb.log(eval_results)

    # Save the model
    if (epoch_i+1) % cluster_config['save_model_every_n_epochs'] == 0:
        model_path = f"models_data/{cluster_config['dir_name']}/model_epoch_{epoch_i+1}.pth"
        os.makedirs(f"models_data/{cluster_config['dir_name']}", exist_ok=True)
        torch.save({
            'eval_results': eval_results,
            'epoch': epoch_i,
            'model_state_dict': model.state_dict(),
            'electrode_embeddings_state_dict': electrode_embeddings.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
if wandb: wandb.finish()