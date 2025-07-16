import shap
import matplotlib.pyplot as plt
import pandas as pd
import sys


import shap

def interpret_model(pipeline, X_test):
    model = pipeline.named_steps['xgb']  

    
    scaler = pipeline.named_steps['scaler']
    X_test_scaled = scaler.transform(X_test)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    shap.summary_plot(shap_values, X_test_scaled, plot_type="dot")
    shap.summary_plot(shap_values, X_test_scaled, plot_type="bar")




if __name__ == "__main__":
    import joblib 

    if len(sys.argv) != 3:
        print("Usage: python model_interpretation.py <model_path> <X_test_csv>")
        sys.exit(1)

    model_path = sys.argv[1]
    X_test_path = sys.argv[2]

 
    model = joblib.load(model_path)

   
    X_test = pd.read_csv(X_test_path)

    interpret_model(model, X_test)
