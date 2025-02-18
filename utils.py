import time
import psutil
import torch
import numpy as np

max_log_priority = 10
def log(message, priority=-1, indent=0):
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
    dir_name += f"_dm{model_config['transformer']['d_model']}"
    #dir_name += f"_ed{model_config['transformer']['embedding_dim']}"
    dir_name += f"_nh{model_config['transformer']['n_heads']}"
    dir_name += f"_nl{model_config['transformer']['n_layers']}"
    dir_name += f"_dr{model_config['transformer']['dropout']}"
    dir_name += f"_bs{training_config['batch_size']}"
    dir_name += f"_wd{training_config['weight_decay']}"
    dir_name += f"_lr{training_config['learning_rate']}"
    dir_name += f"_opt{training_config['optimizer']}"
    dir_name += f"_r{training_config['random_string']}"
    cluster_config['dir_name'] = dir_name
    return dir_name