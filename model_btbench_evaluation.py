from torch.utils.data import DataLoader
import sklearn 
import numpy as np
from btbench_train_test_splits import generate_splits_SS_SM

# Evaluation class for Same Subject Same Movie (SS-SM), on btbench evals
class FrozenModelEvaluation_SS_SM():
    def __init__(self, eval_names, subject_trials, dtype, batch_size,
                 # regression parameters
                 regression_random_state=42,  regression_solver='lbfgs', 
                 regression_tol=1e-3,  regression_n_jobs=5,
                 regression_max_iter=10000):
        """
        Args:
            eval_names (list): List of evaluation metric names to use (e.g. ["volume", "word_gap"])
            subject_trials (list): List of tuples where each tuple contains (subject, trial_id).
                                 subject is a BrainTreebankSubject object and trial_id is an integer.
            dtype (torch.dtype, optional): Data type for tensors.
        """
        self.eval_names = eval_names
        self.subject_trials = subject_trials
        self.all_subject_identifiers = set([subject.subject_identifier for subject, trial_id in self.subject_trials])
        self.dtype = dtype
        self.batch_size = batch_size

        self.regression_max_iter = regression_max_iter
        self.regression_random_state = regression_random_state
        self.regression_solver = regression_solver
        self.regression_tol = regression_tol
        self.regression_n_jobs = regression_n_jobs

        evaluation_datasets = {}
        for eval_name in self.eval_names:
            for subject, trial_id in self.subject_trials:
                splits = generate_splits_SS_SM(subject, trial_id, eval_name, dtype=self.dtype)
                evaluation_datasets[(eval_name, subject.subject_identifier, trial_id)] = splits
        self.evaluation_datasets = evaluation_datasets

    def _evaluate_on_dataset(self, model, electrode_embed, train_dataset, test_dataset, verbose=False):
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        device, dtype = model.device, model.dtype
        X_train, y_train = [], []
        if verbose: print('generating frozen features')
        for i, (batch_input, batch_label) in enumerate(train_dataloader):
            batch_input = batch_input.to(device, dtype=dtype)
            if verbose: print(f'generating frozen features for batch {i} of {len(train_dataloader)}')
            features = model.generate_frozen_evaluation_features(batch_input, electrode_embed).detach().cpu().float().numpy()
            if verbose: print(f'done generating frozen features for batch {i} of {len(train_dataloader)}')
            X_train.append(features)
            y_train.append(batch_label.numpy())

        X_test, y_test = [], []
        for i, (batch_input, batch_label) in enumerate(test_dataloader):
            batch_input = batch_input.to(device, dtype=dtype)
            if verbose: print(f'generating frozen features for batch {i} of {len(test_dataloader)}')
            features = model.generate_frozen_evaluation_features(batch_input, electrode_embed).detach().cpu().float().numpy()
            if verbose: print(f'done generating frozen features for batch {i} of {len(test_dataloader)}')
            X_test.append(features)
            y_test.append(batch_label.numpy())
        if verbose: print('done generating frozen features')

        if verbose: print("creating numpy arrays")
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)
        if verbose: print("done creating numpy arrays")

        regressor = sklearn.linear_model.LogisticRegression(
            random_state=self.regression_random_state, 
            max_iter=self.regression_max_iter, 
            n_jobs=self.regression_n_jobs, 
            solver=self.regression_solver, 
            tol=self.regression_tol
        )

        if verbose: print('fitting regressor')
        regressor.fit(X_train, y_train)
        if verbose: print('done fitting regressor')
        regressor_pred = regressor.predict_proba(X_test)[:, 1]
        regressor_pred_class = regressor.predict(X_test)
        auroc = sklearn.metrics.roc_auc_score(y_test, regressor_pred, multi_class='ovr')
        accuracy = sklearn.metrics.accuracy_score(y_test, regressor_pred_class)
        if verbose: print('done evaluating')
        return auroc, accuracy
    
    def _evaluate_on_metric_cv(self, model, electrode_embed, train_datasets, test_datasets, verbose=False, quick_eval=False):
        auroc_list, accuracy_list = [], []
        for train_dataset, test_dataset in zip(train_datasets, test_datasets):
            auroc, accuracy = self._evaluate_on_dataset(model, electrode_embed, train_dataset, test_dataset, verbose=verbose)
            auroc_list.append(auroc)
            accuracy_list.append(accuracy)
            if quick_eval:
                break
        return np.mean(auroc_list), np.mean(accuracy_list)
    
    def evaluate_on_all_metrics(self, model, electrode_embedding, verbose=False, quick_eval=False):
        evaluation_results = {}
        for subject_identifier in self.all_subject_identifiers:
            electrode_embed = electrode_embedding(subject_identifier).to(model.device, model.dtype)
            for eval_name in self.eval_names:
                trial_ids = [trial_id for subject, trial_id in self.subject_trials if subject.subject_identifier == subject_identifier]
                for trial_id in trial_ids:
                    splits = self.evaluation_datasets[(eval_name, subject_identifier, trial_id)]
                    auroc, accuracy = self._evaluate_on_metric_cv(model, electrode_embed, splits[0], splits[1], verbose=verbose, quick_eval=quick_eval)
                    evaluation_results[(eval_name, subject_identifier, trial_id)] = (auroc, accuracy)
        
        evaluation_results_strings = self._format_evaluation_results_strings(evaluation_results)
        return evaluation_results_strings

    def _format_evaluation_results_strings(self, evaluation_results):
        evaluation_results_strings = {}
        for eval_name in self.eval_names:
            auroc_values = []
            acc_values = []
            subject_aurocs = {}
            subject_accs = {}
            for (metric, subject_identifier, trial_id) in [key for key in evaluation_results.keys() if key[0] == eval_name]:
                if subject_identifier not in subject_aurocs:
                    subject_aurocs[subject_identifier] = []
                    subject_accs[subject_identifier] = []
                auroc, accuracy = evaluation_results[(eval_name, subject_identifier, trial_id)]
                auroc, accuracy = auroc.item(), accuracy.item()

                subject_aurocs[subject_identifier].append(auroc)
                subject_accs[subject_identifier].append(accuracy)
                evaluation_results_strings[f"eval_auroc/{subject_identifier}_{trial_id}_{eval_name}"] = auroc
                evaluation_results_strings[f"eval_acc/{subject_identifier}_{trial_id}_{eval_name}"] = accuracy
            for subject_identifier in subject_aurocs:
                auroc_values.append(np.mean(subject_aurocs[subject_identifier]).item())
                acc_values.append(np.mean(subject_accs[subject_identifier]).item())
            evaluation_results_strings[f"eval_auroc/average_{eval_name}"] = np.mean(auroc_values).item()
            evaluation_results_strings[f"eval_acc/average_{eval_name}"] = np.mean(acc_values).item()
        return evaluation_results_strings