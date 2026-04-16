# Equity Factor Model: Cross-Sectional Return Decomposition

A practical implementation of a **fundamental equity factor model** in Python, inspired by the framework in *The Elements of Quantitative Investing* by Giuseppe A. Paleologo.

This project estimates the cross-sectional model


$\mathbf{r_t} = \boldsymbol{\alpha} + \mathbf{B_t} \mathbf{f_t} + \boldsymbol{\epsilon_t}$

at each month end, where:

- $\mathbf{r_t}$ is the vector of forward stock returns
- $\mathbf{B_t}$ is the matrix of stock factor exposures
- $\mathbf{f_t}$ is the vector of realised factor returns
- $\boldsymbol{\epsilon_t}$ is the vector of idiosyncratic residuals


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
└── requirements.txt
