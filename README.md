# Financial Sentiment-Driven Return Predictor using FinBERT

## Problem Statement
Can NLP-extracted sentiment from financial news headlines improve 
next-day directional return prediction for Nifty 50 beyond 
price-based features alone?

## Methodology
1. Collected Nifty 50 price data (Feb-Aug 2025) using yfinance
2. Scored 3,024 financial headlines using FinBERT (ProsusAI/finbert)
3. Engineered 11 features: sentiment rolling averages (1d/3d/7d), 
   sentiment momentum, headline count, price returns, volatility
4. Benchmarked Logistic Regression vs XGBoost (Optuna-tuned)
5. Evaluated using walk-forward time-series cross-validation (5 folds)
6. Applied SHAP TreeExplainer for feature interpretability
7. Tracked all experiments with MLflow

## Results
| Model | F1 | AUC-ROC | Directional Accuracy |
|---|---|---|---|
| Baseline (always UP) | 0.65 | - | 48.4% |
| Logistic Regression | 0.44 | - | - |
| XGBoost (Optuna tuned) | 0.50 | 0.54 | 60.0% |

## Key Findings
- XGBoost achieved 60% directional accuracy vs 48% baseline
- headline_count was the strongest predictor via SHAP analysis
- 3-day rolling sentiment outperformed single-day sentiment
- Weak linear separation motivated non-linear ML approach

## Limitations
- Dataset limited to 6 months (124 trading days)
- Future: extend to multi-year dataset, real-time news sources

## How to Run
pip install -r requirements.txt
Run notebooks in order: 01 to 05
View experiments: mlflow ui
