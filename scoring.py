def min_max_norm(x):
    """0-1 arasında min-max normalizasiya"""
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

def score_wallets(features_df):
  
    txn_score = min_max_norm(features_df['total_txn_count'])
    deposit_score = min_max_norm(features_df['total_deposit_count'])
    borrow_score = min_max_norm(features_df['total_borrow_count'])
    repay_score = min_max_norm(features_df['total_repay_count'])
    
    liquidation_score = 1 - min_max_norm(features_df['total_liquidation_count'])
    amount_score = min_max_norm(features_df['total_amount_usd'])

    txns_per_day_score = min_max_norm(features_df['txns_per_active_day'])
    liquidation_ratio_score = 1 - min_max_norm(features_df['liquidation_ratio'])
    borrow_repay_diff_score = 1 - min_max_norm(features_df['borrow_repay_diff'].clip(lower=0))
    net_usd_flow_score = min_max_norm(features_df['net_usd_flow'].clip(lower=0))

   
    weights = {
        'txn': 0.15,
        'deposit': 0.1,
        'borrow': 0.1,
        'repay': 0.1,
        'liquidation': 0.15,
        'amount': 0.1,
        'txns_per_day': 0.1,
        'liquidation_ratio': 0.1,
        'borrow_repay_diff': 0.05,
        'net_usd_flow': 0.05,
    }

   
    score = (
        txn_score * weights['txn'] +
        deposit_score * weights['deposit'] +
        borrow_score * weights['borrow'] +
        repay_score * weights['repay'] +
        liquidation_score * weights['liquidation'] +
        amount_score * weights['amount'] +
        txns_per_day_score * weights['txns_per_day'] +
        liquidation_ratio_score * weights['liquidation_ratio'] +
        borrow_repay_diff_score * weights['borrow_repay_diff'] +
        net_usd_flow_score * weights['net_usd_flow']
    )



    return (score * 1000).round().astype(int)
