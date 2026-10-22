# Customer Churn Prediction using Decision Tree Induction
DMDW Micro Project (E1CSA 382), Alliance University.

Predicts telecom customer churn on the IBM Telco dataset (7,043 customers) using
Decision Tree (Gini), Decision Tree (Entropy, tuned and pruned) and Random Forest.

## Run
    pip install -r requirements.txt
    python churn_decision_tree.py

Figures and metrics are written to `outputs/`. Pre-generated results are in `results/`.

## Results (20% test set)
| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Decision Tree (Gini, unpruned) | 0.742 | 0.495 | 0.504 | 0.662 |
| Decision Tree (Entropy, tuned) | 0.750 | 0.767 | 0.620 | 0.835 |
| Random Forest | 0.764 | 0.775 | 0.636 | 0.844 |
