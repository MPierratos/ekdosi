import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from omegaconf import OmegaConf  
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from ekdosi.configs.trainer.config import ExecutorConfig


def print_progress(start_time, progress_percentage):
    elapsed_time = (time.time() - start_time) / 60  # Convert to minutes
    estimated_time = elapsed_time / progress_percentage - elapsed_time
    print(
        f"Elapsed time: {elapsed_time:.2f}m, Estimated time remaining: {estimated_time:.2f}m"
    )


class HDF5Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as file:
            self.features = file["train"]["features"][:]
            self.targets = file["train"]["targets"][:]
            self.class_weights = file["class_weights"][:]  # Load class weights

    def get_data(self):
        return self.features, self.targets, self.class_weights


class XGBoostTrainer:
    def __init__(self, config: ExecutorConfig):
        """
        Initializes the XGBoostTrainer class.

        Args:
            config (Config): The configuration settings.
        """
        self.config = config
        self.dataset = HDF5Dataset(config.dataset.name + ".hdf5")
        self.features, self.targets, self.class_weights = self.dataset.get_data()
        self.kf = KFold(n_splits=self.config.train.k_folds, shuffle=True, random_state=42)
        self.best_models = []
        if self.config.train.feature_selection:
            self.features = self._feature_selection(self.features, self.targets)
  
    def _feature_selection(self, features, targets):
        """
        Perform feature selection before training.

        Args:
            features (np.array): The feature matrix.
            targets (np.array): The target matrix.

        Returns:
            np.array: The reduced feature matrix.
        """
        if self.config.train.feature_selection_method == "xgboost":
            # Use a basic XGBoost model for feature selection
            selection_model = xgb.XGBRegressor()
            selection_model.fit(features, targets)
            selection = SelectFromModel(selection_model, prefit=True)
            return selection.transform(features)
        elif self.config.train.feature_selection_method == "mutual_info":
            # Use mutual information for feature selection
            mi = mutual_info_regression(features, targets)
            mi_threshold = np.median(mi)  # Using median as threshold for feature selection
            selected_features = features[:, mi > mi_threshold]
            return selected_features
        else:
            raise ValueError("Unsupported feature selection method")
  
    def train(self):
        start_time = time.time()
        fold = 0
        evals_results = []
        for train_index, val_index in self.kf.split(self.features):
            X_train, X_val = self.features[train_index], self.features[val_index]
            y_train, y_val = self.targets[train_index], self.targets[val_index]
            class_weights_train = self.class_weights[train_index]
            best_models_fold = []
            evals_results_fold = []
            for target_col in range(y_train.shape[-1]):  # Iterate through each target column
                dtrain = xgb.DMatrix(X_train, label=y_train[:, target_col], weight=class_weights_train)
                dval = xgb.DMatrix(X_val, label=y_val[:, target_col])
                params = {
                    'max_depth': self.config.model.max_depth,
                    'eta': self.config.optimizer.optimizer_extra_configs.lr,
                    'objective': 'reg:squarederror',
                    'nthread': 4,
                    'eval_metric': 'rmse'
                }
                evals_result = {}
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.config.model.num_boost_round,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=self.config.train.early_stopping_rounds,
                    evals_result=evals_result,
                    verbose_eval=True
                )
                best_models_fold.append((model, min(evals_result['eval']['rmse'])))
                evals_results_fold.append(evals_result)
            self.best_models.append(best_models_fold)
            evals_results.append(evals_results_fold)
            fold += 1
            print(f"Finished Training Fold {fold}")
            print_progress(start_time, fold / self.config.train.k_folds)
        print("Finished Training All Folds")
        # Save the best model for each target
        for target_col in range(len(self.best_models[0])):
            best_model = min(self.best_models, key=lambda x: x[target_col][1])[target_col][0]
            best_model.save_model(f"best_model_target_{target_col}.model")

    def _evaluate(self):
        for i, models in enumerate(self.best_models):
            for j, (model, _) in enumerate(models):
                y_pred = model.predict(self.dval)
                mse = mean_squared_error(self.y_val[:, j], y_pred)
                print(f"Validation MSE for Fold {i+1}, Output {j+1}: {mse:.4f}")

def plot_losses(evals_results):
    for i, evals_results_fold in enumerate(evals_results):
        for j, evals_result in enumerate(evals_results_fold):
            epochs = len(evals_result['eval']['rmse'])
            x_axis = range(0, epochs)
            plt.figure(figsize=(10, 5))
            plt.plot(x_axis, evals_result['eval']['rmse'], label=f'RMSE Fold {i+1}, Output {j+1}')
            plt.xlabel("Epochs")
            plt.ylabel("RMSE")
            plt.title(f"XGBoost Training and Validation RMSE for Output {j+1}")
            plt.legend()
            plt.show()


