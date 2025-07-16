import pandas as pd
import numpy as np

def feature_engineer(df):
    features = df.groupby('userWallet').agg(
        total_txn_count=('action', 'count'),
        active_days=('date', 'nunique'),
        first_txn_date=('timestamp', 'min'),
        last_txn_date=('timestamp', 'max'),
        total_deposit_count=('is_deposit', 'sum'),
        total_borrow_count=('is_borrow', 'sum'),
        total_repay_count=('is_repay', 'sum'),
        total_liquidation_count=('is_liquidation', 'sum'),
        total_redeem_count=('is_redeem', 'sum'),
        total_amount_usd=('amount_usd', 'sum'),
        avg_amount_usd=('amount_usd', 'mean'),
        median_amount_usd=('amount_usd', 'median')
    )
    features['days_between_first_last'] = (features['last_txn_date'] - features['first_txn_date']).dt.days

    deposit_usd = df[df['is_deposit'] == 1].groupby('userWallet')['amount_usd'].sum().rename('total_deposit_usd')
    borrow_usd = df[df['is_borrow'] == 1].groupby('userWallet')['amount_usd'].sum().rename('total_borrow_usd')
    repay_usd = df[df['is_repay'] == 1].groupby('userWallet')['amount_usd'].sum().rename('total_repay_usd')

    features = features.join([deposit_usd, borrow_usd, repay_usd]).fillna(0)

    features['borrow_deposit_ratio'] = features['total_borrow_usd'] / (features['total_deposit_usd'] + 1e-6)
    features['repay_borrow_ratio'] = features['total_repay_usd'] / (features['total_borrow_usd'] + 1e-6)
    features['has_liquidation'] = (features['total_liquidation_count'] > 0).astype(int)
    features['txns_per_active_day'] = features['total_txn_count'] / (features['active_days'] + 1e-9)
    features['liquidation_ratio'] = features['total_liquidation_count'] / (features['total_txn_count'] + 1e-9)
    features['borrow_repay_diff'] = features['total_borrow_count'] - features['total_repay_count']
    features['net_usd_flow'] = features['total_deposit_usd'] - features['total_borrow_usd']
    features['active_days_ratio'] = features['active_days'] / (features['days_between_first_last'] + 1e-9)  # aktivlik nisbəti

    return features
