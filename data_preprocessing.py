import pandas as pd
import json

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def preprocess(df):
    if 'actionData' not in df.columns:
        raise ValueError("'actionData' column not found in the dataframe.")

    def extract_from_actionData(row):
        try:
            amount = int(row.get('amount', 0)) / 1e6
        except (ValueError, TypeError):
            amount = 0
        asset = row.get('assetSymbol', 'UNKNOWN')
        try:
            price = float(row.get('assetPriceUSD', 0))
        except (ValueError, TypeError):
            price = 0.0
        type_ = row.get('type', 'UNKNOWN')
        return pd.Series([amount, asset, price, type_])

    df[['amount', 'asset', 'price_usd', 'action_type']] = df['actionData'].apply(extract_from_actionData)
    df['amount_usd'] = df['amount'] * df['price_usd']

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['date'] = df['timestamp'].dt.date
    else:
        raise ValueError("'timestamp' column not found in the dataframe.")

    df['is_deposit'] = (df['action'].str.lower() == 'deposit').astype(int)
    df['is_borrow'] = (df['action'].str.lower() == 'borrow').astype(int)
    df['is_repay'] = (df['action'].str.lower() == 'repay').astype(int)
    df['is_redeem'] = (df['action'].str.lower() == 'redeemunderlying').astype(int)
    df['is_liquidation'] = (df['action'].str.lower() == 'liquidationcall').astype(int)

    return df
