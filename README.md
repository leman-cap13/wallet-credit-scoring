

This project implements a robust and modular pipeline to generate credit scores for wallets interacting with the Aave V2 protocol based on their historical transaction behavior. The codebase is structured into logically separated modules, each responsible for a specific step of the end-to-end process.

Data Loading and Preprocessing
Raw transaction data is ingested from a JSON file, parsed into a structured pandas DataFrame.

Nested fields inside transaction records, such as actionData, are safely extracted with error handling to convert stringified numerical values into usable numeric types.

New columns are engineered to capture transaction context, such as normalized transaction amounts in USD, timestamp conversion to datetime, and binary flags for different action types (deposit, borrow, repay, etc.).

Robust type conversions and missing value handling ensure data quality and consistency before further processing.

Feature Engineering
Transaction records are aggregated on a per-wallet basis, deriving statistical summaries like total transaction count, number of active days, first and last transaction timestamps.

Behavioral metrics including counts of specific transaction types, USD volume sums, average transaction size, and durations between wallet activity extremes are computed.

Derived ratios and flags highlight risk factors and wallet behaviors, such as borrow-to-deposit ratios, repayment efficiency, liquidation occurrence, transaction frequency per active day, and net USD flow.

Careful use of small constants prevents division-by-zero errors, improving numerical stability.

Scoring Function
Wallet features are normalized using min-max scaling to map diverse metrics into a comparable range.

A transparent, interpretable scoring function combines weighted feature contributions to generate a final credit score from 0 to 1000.

Weights reflect the relative importance of transaction volume, repayment behavior, liquidation presence, and other key risk indicators.

Model Preparation and Training
An XGBoost regression model is trained on engineered features to predict credit scores.

A pipeline encapsulates standard scaling and model training for reproducibility and clean workflow.

Data is split into training and testing sets, with evaluation metrics including mean absolute error (MAE) and R² reported.

Feature importance analysis guides model interpretability and future feature selection.

Model Interpretation
SHAP (SHapley Additive exPlanations) values quantify individual feature contributions globally and per wallet, enabling explainable AI insights.

Summary and dependence plots help identify how features influence credit score predictions.

Hyperparameter Tuning
Optuna framework performs efficient hyperparameter optimization for the XGBoost model.

Automated trials maximize validation performance, balancing accuracy and generalization.

Visualization
Score distributions are visualized to understand wallet risk segmentation.

Feature importance and SHAP plots communicate model behavior clearly to stakeholders.

Overall, the code prioritizes clean, maintainable design with modular functions and clear data flow, allowing easy extensions and adaptation for further DeFi credit scoring tasks.


