# The Elements Of Quantitative Investing (TEOQI)

This is my project for implementing ideas in Giuseppe A. Paleologo's red book *The Elements of Quantitative Investing*. I very much enjoyed reading the book and it inspired me to implement some of the ideas and concepts. For me, I always find this is the best way of properly undertanding a complex topic. 

Firstly, I decided to implement an equity factor model from scratch, building 11 different factors as introduced in the notebook 'linear_models_of_returns.ipynb'. To create the factors, the models ingest fundamental company data and market data for around 500 stocks in the S&P 500, over the last fifteen years. The factor matrices created can then be used to estimate an equity factor model, as shown in the notebook.

# Equity Factor Model: Cross-Sectional Return Decomposition
A practical implementation of a fundamental equity factor model in Python, inspired by the framework in *The Elements of Quantitative Investing* by Giuseppe A. Paleologo.

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
