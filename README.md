# The Elements Of Quantitative Investing

This is my project for implementing ideas in Giuseppe A. Paleologo's red book The Elements of Quantitative Investing.

Firstly, I have implemented an equity factor model from scratch, building 11 different factors as introduced in the notebook:

[linear models of returns](linear_models_of_returns.ipynb)

To create the factors, the models ingest fundamental company data and market data for the stocks in the S&P 500, over the last fifteen years.

The factor matrices created can then be used to estimate the equity factor model

$\mathbf{r_t} = \boldsymbol{\alpha} + \mathbf{B_t} \mathbf{f_t} + \boldsymbol{\epsilon_t}$

at each month end, where:

- $\mathbf{r_t}$ is the vector of forward stock returns
- $\mathbf{B_t}$ is the matrix of stock factor exposures
- $\mathbf{f_t}$ is the vector of realised factor returns
- $\boldsymbol{\epsilon_t}$ is the vector of idiosyncratic residuals

