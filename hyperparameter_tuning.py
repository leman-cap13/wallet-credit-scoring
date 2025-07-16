import optuna
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score

def tune_hyperparameters(features):
    numeric_features = [
        'total_txn_count', 'active_days', 'total_deposit_count',
        'total_borrow_count', 'total_repay_count', 'total_liquidation_count',
        'total_redeem_count', 'total_amount_usd', 'avg_amount_usd',
        'days_between_first_last', 'total_deposit_usd', 'total_borrow_usd',
        'total_repay_usd', 'borrow_deposit_ratio', 'repay_borrow_ratio',
        'has_liquidation', 'txns_per_active_day', 'liquidation_ratio',
        'borrow_repay_diff', 'net_usd_flow'
    ]

    X = features[numeric_features]  
    y = features['credit_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state': 42,
            'verbosity': 0,
            'objective': 'reg:squarederror'
        }
        model = XGBRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Best params:", study.best_params)
    print("Best CV R2 score:", study.best_value)

    return study.best_params



