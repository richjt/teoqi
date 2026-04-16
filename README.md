# Equity Factor Model: Cross-Sectional Return Decomposition

A practical implementation of a **fundamental equity factor model** in Python, inspired by the framework in *The Elements of Quantitative Investing* by Giuseppe A. Paleologo.

This project estimates the cross-sectional model

\[
r_t = B_t f_t + \epsilon_t
\]

at each month end, where:

- `r_t` is the vector of forward stock returns
- `B_t` is the matrix of stock factor exposures
- `f_t` is the vector of realised factor returns
- `\epsilon_t` is the vector of idiosyncratic residuals


## Repository structure

```text
.
├──linear_models_of_returns.ipynb
├── python/
│   ├── data_pipeline.py
│   ├── factor_builder.py
├── factor_data/
│   ├── factor_data.csv
│   ├── sectors_clean.py
│   ├── ticker_list.py
├── README.md
└── requirements.txt# My New Project
