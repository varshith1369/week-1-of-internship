# Week 1 – Data Exploration and Problem Definition (Diamonds Dataset)

Exploratory data analysis of the public Diamonds dataset (53,940 rows, 10 columns) using Python.

## Contents
- `eda.py` – EDA script: structure, missing values, duplicates, anomalies, summary statistics, correlations, ANOVA / Kruskal-Wallis tests, and figure generation
- `diamonds.csv` – dataset (ggplot2 diamonds, via the seaborn-data repository)
- `figures/` – generated plots
- `Week1_Data_Exploration_Report.docx` – full report with problem definition and analysis plan

## Run
```
pip install -r requirements.txt
python eda.py
```

## Problem statement
Identify which characteristics (carat, cut, colour, clarity, depth, table, dimensions) drive diamond price, quantify each effect after controlling for carat, and build a predictive model of price.

## Key findings
- No missing values; 146 duplicate rows; 20 rows with zero dimensions; 3 extreme outliers
- Price is right-skewed (1.62) and near-symmetric after log transform (0.12)
- Carat vs price correlation is 0.92; carat confounds cut, colour and clarity effects
