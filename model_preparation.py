import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def prepare_and_train_model(features):
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

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=1
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    model = pipeline.named_steps['xgb']
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Gain)")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.show()

    
    return pipeline, X_test, y_test 
