import matplotlib.pyplot as plt
import seaborn as sns

def plot_credit_score_distribution(credit_score_series):
    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    sns.histplot(credit_score_series, bins=50, kde=True)
    plt.title('Credit Score Distribution (Histogram)')
    plt.xlabel('Credit Score')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=credit_score_series)
    plt.title('Credit Score Distribution (Boxplot)')
    plt.xlabel('Credit Score')

    plt.tight_layout()
    plt.show()


def plot_score_vs_features(features):
    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(data=features, x='total_txn_count', y='credit_score')
    plt.title('Credit Score vs Total Transaction Count')
    plt.xlabel('Total Transaction Count')
    plt.ylabel('Credit Score')

    plt.subplot(1, 2, 2)
    sns.scatterplot(data=features, x='total_liquidation_count', y='credit_score')
    plt.title('Credit Score vs Total Liquidation Count')
    plt.xlabel('Total Liquidation Count')
    plt.ylabel('Credit Score')

    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(features):
    corr = features[['credit_score', 'total_txn_count', 'total_liquidation_count']].corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix Heatmap')
    plt.show()
