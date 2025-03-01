import time
import psutil
import torch
import numpy as np
import argparse


def parse_configs_from_args(training_config, model_config, cluster_config):
    parser = argparse.ArgumentParser()
    
    # Transformer model arguments
    parser.add_argument('--d_model', type=int, default=None, help='Dimension of transformer model')
    parser.add_argument('--n_heads', type=int, default=None, help='Number of attention heads')
    parser.add_argument('--n_layers_electrode', type=int, default=None, help='Number of transformer layers for electrode path')
    parser.add_argument('--n_layers_time', type=int, default=None, help='Number of transformer layers for time path')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--embedding_dim', type=int, default=None, help='Dimension of electrode embeddings')
    parser.add_argument('--max_frequency_bin', type=int, default=None, help='Maximum frequency bin')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default=None, help='Optimizer type')
    parser.add_argument('--p_electrodes_per_stream', type=float, default=None, help='Proportion of electrodes per stream')
    
    # Other model config
    parser.add_argument('--init_normalization', type=int, default=None, help='Whether to use initial normalization')

    # Electrode embedding config
    parser.add_argument('--electrode_embedding_type', type=str, default=None, help='Type of electrode embedding')
    parser.add_argument('--electrode_embedding_coordinate_noise_std', type=float, default=None, help='Coordinate noise std for electrode embedding')

    parser.add_argument('--cache_subjects', type=int, default=None, help='Whether to cache subjects')
    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name')
    parser.add_argument('--random_string', type=str, default=None, help='Random string for seed generation')
    

    args = parser.parse_args()
    
    # Update configs with command line args if provided
    if args.d_model is not None:
        model_config['transformer']['d_model'] = args.d_model
    if args.n_heads is not None:
        model_config['transformer']['n_heads'] = args.n_heads
    if args.n_layers_electrode is not None:
        model_config['transformer']['n_layers_electrode'] = args.n_layers_electrode
    if args.n_layers_time is not None:
        model_config['transformer']['n_layers_time'] = args.n_layers_time
    if args.dropout is not None:
        model_config['transformer']['dropout'] = args.dropout
    if args.batch_size is not None:
        training_config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        training_config['learning_rate'] = args.learning_rate
    if args.weight_decay is not None:
        training_config['weight_decay'] = args.weight_decay
    if args.optimizer is not None:
        training_config['optimizer'] = args.optimizer
    if args.p_electrodes_per_stream is not None:
        training_config['p_electrodes_per_stream'] = args.p_electrodes_per_stream
    if args.init_normalization is not None:
        model_config['init_normalization'] = bool(args.init_normalization)
    if args.cache_subjects is not None:
        cluster_config['cache_subjects'] = bool(args.cache_subjects)
    if args.random_string is not None:
        training_config['random_string'] = args.random_string
    if args.electrode_embedding_type is not None:
        model_config['electrode_embedding']['type'] = args.electrode_embedding_type
    if args.electrode_embedding_coordinate_noise_std is not None:
        model_config['electrode_embedding']['coordinate_noise_std'] = args.electrode_embedding_coordinate_noise_std
    if args.wandb_project is not None:
        cluster_config['wandb_project'] = args.wandb_project
    if args.embedding_dim is not None:
        model_config['electrode_embedding']['embedding_dim'] = args.embedding_dim
    if args.max_frequency_bin is not None:
        model_config['max_frequency_bin'] = args.max_frequency_bin


max_log_priority = 1
def log(message, priority=0, indent=0):
    if priority > max_log_priority: return

    current_time = time.strftime("%H:%M:%S")
    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(f"[{current_time} gpu {gpu_memory_reserved:.1f}G ram {ram_usage:.1f}G] ({priority}) {' '*4*indent}{message}")


def update_random_seed(training_config):
    random_seed = hash(training_config['random_string']) % (2**32)
    training_config['random_seed'] = random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    return random_seed


def update_dir_name(model_config, training_config, cluster_config):
    dir_name = "M"
    #dir_name += f"_t{model_config['max_n_timebins']}"
    #dir_name += f"_st{model_config['sample_timebin_size']}"
    dir_name += f"_nst{len(training_config['train_subject_trials'])}"
    if not cluster_config['cache_subjects']:
        dir_name += f"_nCS"
    if model_config['init_normalization']:
        dir_name += f"_iN"
        
    if model_config['electrode_embedding']['type'] == 'coordinate_init':
        dir_name += f"_eeCI"
    elif model_config['electrode_embedding']['type'] == 'noisy_coordinate':
        dir_name += f"_eeNC_ecns{model_config['electrode_embedding']['coordinate_noise_std']}"
    elif model_config['electrode_embedding']['type'] == 'learned':
        dir_name += f""

    if 'p_electrodes_per_stream' in training_config and training_config['p_electrodes_per_stream'] != 0.5:
        dir_name += f"_pps{training_config['p_electrodes_per_stream']}"
    dir_name += f"_dm{model_config['transformer']['d_model']}"
    if model_config['electrode_embedding']['embedding_dim'] is not None:
        dir_name += f"_ed{model_config['electrode_embedding']['embedding_dim']}"
    if model_config['max_frequency_bin'] != 64:
        dir_name += f"_mbf{model_config['max_frequency_bin']}"
    dir_name += f"_nh{model_config['transformer']['n_heads']}"
    dir_name += f"_nl{model_config['transformer']['n_layers_electrode']}" + f"_{model_config['transformer']['n_layers_time']}"
    dir_name += f"_dr{model_config['transformer']['dropout']}"
    dir_name += f"_bs{training_config['batch_size']}"
    dir_name += f"_wd{training_config['weight_decay']}"
    dir_name += f"_lr{training_config['learning_rate']}"
    dir_name += f"_opt{training_config['optimizer']}"
    dir_name += f"_r{training_config['random_string']}"
    cluster_config['dir_name'] = dir_name
    return dir_name