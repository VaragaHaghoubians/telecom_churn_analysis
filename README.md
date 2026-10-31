# Telecom Customer Churn Analysis

A data mining project from my MSc in Data Science at the University of Turin. I used the Iranian Telecom Churn dataset (UCI) and looked at customers in three ways: grouping them, finding unusual ones, and predicting who will leave.

## What I did

1. Clustering with K-Means (k=3) on standardized features, shown in 2D with PCA
2. Anomaly detection with Local Outlier Factor (LOF), to flag customers with unusual usage
3. Churn prediction with a depth-5 decision tree, 80/20 train/test split

## Results

From the saved notebook run:

| Metric | Value |
|---|---|
| Test accuracy | 91.4% (576/630) |
| Churn recall | 64.6% (64/99) |
| Churn precision | 77% |

Only about 16% of customers in the test set churn, so always predicting "no churn" would already give about 84% accuracy. That is why I look at churn recall and precision, not only accuracy. The model finds about 2 out of 3 churners.

## Files

- `telecom_churn_analysis.ipynb`, the notebook with all the code
- `telecom_churn_report.pdf`, the written report

## How to run

```bash
git clone https://github.com/VaragaHaghoubians/telecom_churn_analysis.git
cd telecom_churn_analysis
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

The dataset is not included. Download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset), save it as `Customer Churn.csv` in the same folder as the notebook, then run all cells.

## What I would improve next

- Compare the decision tree with logistic regression and random forest
- Tune the decision threshold to catch more churners
- Use cross-validation instead of a single split

## Author

Varaga Haghoubians, MSc Stochastics & Data Science student, University of Turin, Industrial Engineering background
[LinkedIn](https://www.linkedin.com/in/varagahaghoubians) · [varaga.haghoubians@gmail.com](mailto:varaga.haghoubians@gmail.com)
