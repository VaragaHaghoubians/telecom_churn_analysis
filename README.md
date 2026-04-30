# 📉 Telecom Customer Churn Analysis

> A data mining project applying clustering, anomaly detection, and classification
> to the Iranian Telecom Churn dataset (UCI Machine Learning Repository).

---

## 🎯 Problem

Telecom companies lose revenue when customers churn without warning.
This project explores customer behaviour through **three complementary lenses**:
unsupervised clustering, anomaly detection, and supervised classification —
to understand who churns, who behaves unusually, and how to predict churn.

---

## 🔍 What's Inside

### 1. 🔵 Clustering — K-Means
- Standardised features with `StandardScaler`
- Applied **K-Means** (k=3) to segment customers into behavioural groups
- Visualised clusters using **PCA** (2D projection)

### 2. 🔴 Anomaly Detection — Local Outlier Factor (LOF)
- Applied **LOF** to detect customers with unusual usage patterns
- Visualised outliers vs. inliers in PCA-reduced 2D space
- Flagged anomalous records for further inspection

### 3. 🟢 Classification — Decision Tree
- Target: `Churn` (binary)
- Train/test split (80/20, random_state=42)
- **Decision Tree Classifier** with full evaluation:
  - Confusion Matrix (visualised with `ConfusionMatrixDisplay`)
  - Classification Report (precision, recall, F1)

---

## 📁 Project Structure

```
telecom_churn_analysis/
│
├── telecom_churn_analysis/
│   ├── Varaga_HAghoubians_DM_Project_2025.ipynb   ← main notebook
│   └── Varaga_Haghoubians_Data_minig_project.pdf  ← full written report
└── README.md
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

**Methods:** K-Means · PCA · Local Outlier Factor · Decision Tree · Confusion Matrix · Classification Report

---

## 🚀 How to Run

```bash
git clone https://github.com/VaragaHaghoubians/telecom_churn_analysis.git
cd telecom_churn_analysis
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook
```

Open `Varaga_HAghoubians_DM_Project_2025.ipynb` and run all cells.

> **Dataset:** Iranian Telecom Churn — available from the
> [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Iranian+Churn+Dataset).
> Place `Customer Churn.csv` in the same folder as the notebook.

---

## 👤 Author

**Varaga Haghoubians** — Junior ML/AI Engineer & Data Analyst
[LinkedIn](https://www.linkedin.com/in/varagahaghoubians) · [GitHub](https://github.com/VaragaHaghoubians) · [varaga.haghoubians@gmail.com](mailto:varaga.haghoubians@gmail.com)
