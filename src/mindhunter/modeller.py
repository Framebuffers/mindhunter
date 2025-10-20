from .mindhunter import StatFrame

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import\
    (confusion_matrix,\
    accuracy_score,\
    classification_report,\
    roc_auc_score,\
    mean_squared_error,\
    r2_score)

class StatModel:
    def __init__(self, sf: StatFrame):
        self.da = sf
        
        self.X_train = 0
        self.X_test = 0

        self.y_train = 0
        self.y_test = 0
        self.y_pred = 0
        self.y_pred_proba = 0

        self._results = {}
    
    def train_linear_model(
        self,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 69
        ) -> List[pd.DataFrame | pd.Series | pd.Series]:

        if target_col not in self.da._df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        X = self.da._df.drop(target_col, axis=1)
        y = self.da._df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        coefficients = model.coef_.tolist(),
        feature_names = X.columns.tolist(),

        result = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'intercept': model.intercept_,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_train': y_train,
            'X_train': X_train
        }

        self.X_train = X_train
        self.X_test = X_test
        
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_train = y_train
        
        return [pd.DataFrame.from_dict(result),\
                pd.Series(coefficients),\
                pd.Series(feature_names)]

    def train_logistic_model(
        self,
        target_col: str,
        categorical_cols: list[str] = None, # pyright: ignore[reportArgumentType]
        test_size: float = 0.2,
        random_state: int = 69,
        class_weight: str = None, # type: ignore
        max_iter: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:

        data = self.da._df

        if categorical_cols:
            data = pd.get_dummies(
                data=data, 
                columns=categorical_cols, 
                drop_first=True
            ).astype(int)

        # check this, remove the true value
        target_encoded = f"{target_col}_True" if categorical_cols and target_col in categorical_cols else target_col

        if target_encoded not in data.columns:
            raise ValueError(f"Target column '{target_encoded}' not found in DataFrame")

        X = data.drop(target_encoded, axis=1)
        y = data[target_encoded]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )

        model = LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = print(classification_report(y_test, y_pred, output_dict=True))

        results = pd.DataFrame.from_dict({
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'intercept': model.intercept_[0],
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'classification_report': report,
            'feature_names': X.columns.tolist(),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'X_train': X_train,
            'y_train': y_train
        })
        
        coefficients = pd.Series(model.coef_[0].tolist())
        
        self.X_train = X_train
        self.X_test = X_test
        
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_train = y_train
        self.y_pred_proba = model.predict_proba(X_test)

        return (results, coefficients)
        
    def evaluate_classification_model(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        X_train: pd.DataFrame,
        target_col: str = None, # type: ignore
        class_labels: dict = None # type: ignore
    ) -> List[pd.DataFrame | pd.DataFrame | pd.DataFrame]:
    
        if class_labels is None:
            class_labels = {0: 'negative', 1: 'positive'}
        
        class_dist_train = y_train.value_counts()
        class_dist_test = y_test.value_counts()
        imbalance_ratio = max(class_dist_train) / min(class_dist_train)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        if roc_auc < 0.7:
            roc_auc_quality = 'poor'
        elif roc_auc < 0.8:
            roc_auc_quality = 'acceptable'
        else:
            roc_auc_quality = 'good'
        
        unique_preds = np.unique(y_pred)
        predicts_majority = len(unique_preds) == 1
        
        hit = tp + tn
        miss = fp + fn
        ratio = hit / miss if miss > 0 else float('inf')
        baseline_acc = max(class_dist_test) / len(y_test)
        model_acc = accuracy_score(y_test, y_pred)
        
        leakage = False
        if target_col:
            leakage = target_col in X_train.columns or f'{target_col}_False' in X_train.columns
        
        class_imbalance_training = {
                class_labels[0]: int(class_dist_train.get(0, 0)),
                class_labels[1]: int(class_dist_train.get(1, 0)),
                f'{class_labels[0]}_percentage': round(class_dist_train.get(0, 0) / len(y_train) * 100, 1),
                f'{class_labels[1]}_percentage': round(class_dist_train.get(1, 0) / len(y_train) * 100, 1)
            }
        
        class_imbalance_test =  {
                class_labels[0]: int(class_dist_test.get(0, 0)),
                class_labels[1]: int(class_dist_test.get(1, 0)),
                f'{class_labels[0]}_percentage': round(class_dist_test.get(0, 0) / len(y_test) * 100, 1),
                f'{class_labels[1]}_percentage': round(class_dist_test.get(1, 0) / len(y_test) * 100, 1)
            }
        
        results = {
            'imbalance_ratio': round(imbalance_ratio, 2),
            'is_imbalanced': imbalance_ratio > 3,
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'sensitivity': round(sensitivity, 3),
            'specificity': round(specificity, 3),
            'roc_auc': round(roc_auc, 3),
            'roc_auc_quality': roc_auc_quality,
            'always_predicts_majority': predicts_majority,
            'baseline_accuracy': round(baseline_acc, 3),
            'model_accuracy': round(model_acc, 3),
            'model_barely_above_baseline': model_acc <= baseline_acc + 0.05,
            'leakage': leakage,
            'correct_predictions': int(hit),
            'incorrect_predictions': int(miss),
            'hit_miss_ratio': round(ratio, 2)
        }
        
        return [pd.DataFrame.from_dict(results),\
                pd.DataFrame.from_dict(class_imbalance_training),\
                pd.DataFrame.from_dict(class_imbalance_test)]