
from data_preprocessing import load_data, preprocess
from feature_engineering import feature_engineer
from scoring import score_wallets
from visualization import plot_credit_score_distribution
from model_preparation import prepare_and_train_model
from model_interpretation import interpret_model
from hyperparameter_tuning import tune_hyperparameters

def main():
   
    df_raw = load_data(r'C:\Users\User\Downloads\user-wallet-transactions.json')



    
    df_processed = preprocess(df_raw)

    
    features = feature_engineer(df_processed)

    
    features['credit_score'] = score_wallets(features)

  


    plot_credit_score_distribution(features['credit_score'])

   
    pipeline, X_test,y_test = prepare_and_train_model(features)

   
    interpret_model(pipeline, X_test)


    best_params = tune_hyperparameters(features)
    print("Best params:", best_params)


if __name__ == '__main__':
    main()
