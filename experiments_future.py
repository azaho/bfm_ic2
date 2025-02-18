"""
hwy so i wanna run many A/B experiments where i change smth in the code and see how it impacts performance, and i want those to be eventually supplementary figures in a paper, but thing is i want it to be elegant with how i structure my code and i dont want to have too many oarameters in settings and too many if statements that check for things which became irrelevant after one test. how do i do this elegantly
"""



class ExperimentConfig:
    def __init__(self, name, description, model_variant, training_config, model_config):
        self.name = name
        self.description = description  # For paper documentation
        self.model_variant = model_variant
        self.training_config = training_config
        self.model_config = model_config

class ModelVariant:
    """Base class for different model architectures/approaches"""
    def create_model(self, device, dtype):
        raise NotImplementedError
        
    def create_electrode_embeddings(self, device, dtype):
        raise NotImplementedError

class BaselineVariant(ModelVariant):
    def create_model(self, device, dtype):
        return LinearModel(model_config['d_model'], 
                         model_config['sample_timebin_size']).to(device, dtype=dtype)
    
    def create_electrode_embeddings(self, device, dtype):
        embeddings = ElectrodeEmbeddings_LinearModel(
            model_config['d_model'], 
            model_config['sample_timebin_size']).to(device, dtype=dtype)
        return embeddings

class BFMVariant(ModelVariant):
    def create_model(self, device, dtype):
        return BFMModel_Scuffed(model_config['d_model'], 
                               model_config['sample_timebin_size']).to(device, dtype=dtype)
    
    def create_electrode_embeddings(self, device, dtype):
        embeddings = ElectrodeEmbeddings_Learned(model_config['d_model']).to(device, dtype=dtype)
        return embeddings

# Define experiments
experiments = [
    ExperimentConfig(
        name="baseline_linear",
        description="Baseline model using linear embeddings",
        model_variant=BaselineVariant(),
        training_config={
            'batch_size': 100,
            'learning_rate': 0.0015,
            # ... other config
        },
        model_config={
            'd_model': 192,
            # ... other config
        }
    ),
    ExperimentConfig(
        name="bfm_learned",
        description="BFM model with learned embeddings",
        model_variant=BFMVariant(),
        training_config={
            'batch_size': 100,
            'learning_rate': 0.002,
            # ... different config
        },
        model_config={
            'd_model': 256,
            # ... different config
        }
    ),
]

# Run experiments
for experiment in experiments:
    print(f"Running experiment: {experiment.name}")
    wandb.init(project=training_config['wandb_project'], 
               name=experiment.name,
               config={
                   "description": experiment.description,
                   "training_config": experiment.training_config,
                   "model_config": experiment.model_config
               })
    
    # Create model and embeddings for this variant
    model = experiment.model_variant.create_model(device, dtype)
    electrode_embeddings = experiment.model_variant.create_electrode_embeddings(device, dtype)
    
    # Rest of your training loop...





class BaseExperiment:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        
    def get_model_config(self):
        return {
            'd_model': 192,
            'sample_timebin_size': 256,
            'max_n_timebins': 24,
        }
    
    def get_training_config(self):
        return {
            'batch_size': 100,
            'n_epochs': 200,
            'learning_rate': 0.0015,
            'weight_decay': 0.0,
            'save_model_every_n_epochs': 1,
            'wandb_project': 'bfm_ic2_0',
            'optimizer': 'Muon',
            'p_test': 0.1,
            'train_subject_trials': train_subject_trials,
            'eval_subject_trials': eval_subject_trials,
            'cache_subjects': True,
        }
    
    def create_model(self):
        config = self.get_model_config()
        return BFMModel_Scuffed(
            config['d_model'], 
            config['sample_timebin_size']
        ).to(self.device, dtype=self.dtype)
    
    def create_electrode_embeddings(self):
        config = self.get_model_config()
        embeddings = ElectrodeEmbeddings_Learned(config['d_model']).to(self.device, dtype=self.dtype)
        for subject_identifier in all_subject_identifiers:
            embeddings.add_embedding(
                subject_identifier, 
                all_subjects[subject_identifier].get_n_electrodes(), 
                requires_grad=True
            )
        return embeddings
    
    def create_optimizers(self, model, electrode_embeddings):
        config = self.get_training_config()
        all_params = list(model.parameters()) + list(electrode_embeddings.parameters())
        
        if config['optimizer'] == 'Muon':
            matrix_params = [p for p in all_params if p.ndim >= 2]
            other_params = [p for p in all_params if p.ndim < 2]
            return [
                Muon(matrix_params, lr=config['learning_rate'], momentum=0.95, 
                     nesterov=True, backend='newtonschulz5', backend_steps=5),
                torch.optim.Adam(other_params, lr=config['learning_rate'], 
                               weight_decay=config['weight_decay'])
            ]
        return [
            torch.optim.Adam(all_params, lr=config['learning_rate'], 
                           weight_decay=config['weight_decay'])
        ]

# Example variations:
class LinearModelExperiment(BaseExperiment):
    def create_model(self):
        config = self.get_model_config()
        return LinearModel(
            config['d_model'], 
            config['sample_timebin_size']
        ).to(self.device, dtype=self.dtype)
    
    def create_electrode_embeddings(self):
        config = self.get_model_config()
        embeddings = ElectrodeEmbeddings_LinearModel(
            config['d_model'], 
            config['sample_timebin_size']
        ).to(self.device, dtype=self.dtype)
        # ... rest of setup
        return embeddings

class HigherDimensionalExperiment(BaseExperiment):
    def get_model_config(self):
        config = super().get_model_config()
        config['d_model'] = 384  # Double the dimension
        return config

class FasterLearningExperiment(BaseExperiment):
    def get_training_config(self):
        config = super().get_training_config()
        config['learning_rate'] = 0.003  # Double learning rate
        config['optimizer'] = 'Adam'  # Change optimizer
        return config

# Usage:
def run_experiment(experiment_class):
    experiment = experiment_class()
    model = experiment.create_model()
    electrode_embeddings = experiment.create_electrode_embeddings()
    optimizers = experiment.create_optimizers(model, electrode_embeddings)
    
    # Training loop using experiment.get_training_config() etc.
    # ... rest of your training code ...

# Run experiments
experiments = [
    BaseExperiment,
    LinearModelExperiment,
    HigherDimensionalExperiment,
    FasterLearningExperiment,
]

for experiment_class in experiments:
    print(f"Running experiment: {experiment_class.__name__}")
    run_experiment(experiment_class)