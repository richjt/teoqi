# teoqi — The Elements of Quantitative Investing

An implementation study of equity factor modelling methods from
Giuseppe A. Paleologo's *The Elements of Quantitative Investing*,
applied to S&P 500 constituents over 2010–2026.

The project builds a 10-factor plus sector equity risk model from
fundamental and market data, fits the factor and idiosyncratic
covariance matrices via OLS regression and documents diagnostics on the resulting estimates.

Implemented in Python with Jupyter notebooks for exposition.
[linear models of returns](linear_models_of_returns.ipynb)

To create the factors, the models ingest fundamental company data and market data for the stocks in the S&P 500, over the last fifteen years.

The factor matrices created can then be used to estimate the equity factor model

$\mathbf{r_t} = \boldsymbol{\alpha} + \mathbf{B_t} \mathbf{f_t} + \boldsymbol{\epsilon_t}$

at each month end, where:

- $\mathbf{r_t}$ is the vector of forward stock returns
- $\mathbf{B_t}$ is the matrix of stock factor exposures
- $\mathbf{f_t}$ is the vector of realised factor returns
- $\boldsymbol{\epsilon_t}$ is the vector of idiosyncratic residuals

## Progress

<u>Complete</u>
1. &#9989; Sourced and ingested fundamental equity data and market data to create 727,876 equity factor data points across 10 sylistic factors for 2010-2026, together with sector data. See [data pipleline](./python/data_pipeline.py). 
2. &#9989; Implemented a utility class to create z-scored panel data (the $\mathbf{B}$'s) using the data from the previous step for a given point in time. See [FactorBuilder class](./python/factor_builder.py)
3. &#9989; Used panel data and stock return data to fit $\mathbf{r_t} = \boldsymbol{\alpha} + \mathbf{B} \mathbf{f_t} + \boldsymbol{\epsilon_t}$ for a point in time see [linear models of returns](linear_models_of_returns.ipynb)
4. &#9989; Ran Ordinary Least Squares (OLS) regression over 60 months to obtain initial estimates of the factor covariance matrix and the idiosyncratic covariance matrix. see [linear models of returns](linear_models_of_returns.ipynb)
5. &#9989; Applied Ledoit-Wolf shrinkage to factor covariance matrix:
$$\boldsymbol{{\hat{\Omega}_{f,shrink}}}(\rho) = (1-\rho)\boldsymbol{\hat{\Omega}_f} + \rho\frac{trace(\boldsymbol{\hat{\Omega}_f})}{m}\mathbf{I_m}$$
6. &#9989; Applied diagonalisation to idiosyncratic covariance matrix see [linear models of returns](linear_models_of_returns.ipynb)
7. &#9989; Ran diagnostics on final estimates of estimated covariance matrices and identified an issue with the WLS regression see [linear models of returns](linear_models_of_returns.ipynb)