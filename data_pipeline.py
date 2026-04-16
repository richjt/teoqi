from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf



########################################################
# Data Preparation
########################################################

def clean_daily_data(daily_data: pd.DataFrame,
                     tickers: list[str]) -> pd.DataFrame: 
    """
    daily_data is a dataframe with a column 'Date' and a column 'ticker' and a column 'measure_name' and a column 'measure_value'.
    tickers is a list of tickers to keep.
    Returns a dataframe with the date column set as the index, the tickers column set as the columns, and the measure_name column set as the columns.
    """
    daily_data["Date"] = pd.to_datetime(daily_data["Date"])
    daily_data["date"] = daily_data["Date"]
    daily_data = daily_data.drop(columns=["Date"])
    daily_data = daily_data.set_index("date")
    daily_data = daily_data.copy()[tickers]

    return daily_data

def to_long_format(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide monthly panel back to long format:
    date | ticker | measure_name | measure_value
    """
    long_df = (
        monthly_panel
        .stack()  # moves columns -> rows
        .rename("measure_value")
        .reset_index()
        .rename(columns={"level_2": "measure_name"})
    )

    return long_df

def prepare_monthly_accounting_data(
    accounting_df: pd.DataFrame,
    lag_months: int=3,
    monthly_freq: str="ME",
    ) -> pd.DataFrame:
    """
    Prepare the monthly accounting data by converting the data to a long format.
    accounting_df is a dataframe with a column 'date' and a column 'ticker' and a column 'measure_name' and a column 'measure_value'.
    lag_months is the number of months to lag the data.
    monthly_freq is the frequency of the data.
    Returns a dataframe with the date column set as the index, the tickers column set as the columns, and the measure_name column set as the columns.
    The measure_value column is the value of the measure.
    The date column is the date of the data.
    The tickers column is the tickers of the data.
    The measure_name column is the name of the measure.
    The measure_value column is the value of the measure.
    """

    df = accounting_df.copy()

    df["date"] = pd.to_datetime(df["date"])

    wide = (
        df.pivot_table(
            index=["date", "ticker"],
            columns="measure_name",
            values="measure_value",
            aggfunc="last"
        ).sort_index()
    )

    # remove the multi-level column index
    wide.columns.name = None

    # reset the index to make "date" and "ticker" columns
    wide = wide.reset_index()

    # shift the date by the specified number of months
    wide["date"] = wide["date"] + pd.DateOffset(months=lag_months)

    # snap lagged dates to month-end
    wide["date"] = wide["date"] + pd.offsets.MonthEnd(0)

    # rebuild as indexed panel
    wide = wide.set_index(["date", "ticker"]).sort_index()

    all_dates = wide.index.get_level_values("date")
    panel_start = all_dates.min()
    panel_end = all_dates.max()

    panel_start = panel_start + pd.offsets.MonthEnd(0)
    panel_end = panel_end + pd.offsets.MonthEnd(0)

    monthly_dates = pd.date_range(panel_start, panel_end, freq=monthly_freq)
    tickers = wide.index.get_level_values("ticker").unique()

    full_index = pd.MultiIndex.from_product(
        [monthly_dates, tickers],
        names=["date", "ticker"]
    )

    # expand to full monthly grid and carry forward within each ticker
    monthly_panel = (
        wide.reindex(full_index)
        .groupby("ticker", group_keys=False)
        .ffill()
        .sort_index()
    )

    return to_long_format(monthly_panel)

def prepare_monthly_market_cap(market_cap_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-form market cap data to monthly month-end long-form data.

    Expected input columns:
        date | ticker | measure_name | measure_value

    where measure_name == 'MarketCap'
    """
    
    df = market_cap_long.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df.loc[df["measure_name"] == "MarketCap"].copy()
    if df.empty:
        raise ValueError("No rows found with measure_name == 'MarketCap'.")

    df = df.sort_values(["ticker", "date"])

    monthly = (
        df.groupby(["ticker", pd.Grouper(key="date", freq="ME")], as_index=False)["measure_value"]
        .last()
    )

    monthly["measure_name"] = "MarketCap"
    monthly = monthly[["date", "ticker", "measure_name", "measure_value"]]

    return monthly

########################################################
# Factor Calculations
########################################################
def calculate_profitability(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate operating profitability factor and return long format:

        date | ticker | measure_name | measure_value

    Uses:
        OperatingProfitability =
            (TotalRevenue
             - CostOfRevenue
             - SellingGeneralAndAdministration
             - InterestExpense_or_proxy)
            / StockholdersEquity

    Notes
    -----
    - Missing expense fields are set to 0 where appropriate.
    - StockholdersEquity must be positive.
    - Assumes the input data has already been time-aligned / lagged upstream.
    """
    required_cols = {"date", "ticker", "measure_name", "measure_value"}
    missing_cols = required_cols - set(df_long.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df = df_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    df_wide = (
        df.pivot_table(
            index=["date", "ticker"],
            columns="measure_name",
            values="measure_value",
            aggfunc="last",
        )
        .reset_index()
        .sort_values(["ticker", "date"])
    )

    # Required core fields
    core_required = ["TotalRevenue", "StockholdersEquity"]
    missing_core = [col for col in core_required if col not in df_wide.columns]
    if missing_core:
        raise ValueError(f"Missing required measures after pivot: {missing_core}")

    # Expense fields that are often missing and can reasonably default to 0
    for col in ["CostOfRevenue", "SellingGeneralAndAdministration", "InterestExpense"]:
        if col in df_wide.columns:
            df_wide[col] = df_wide[col].fillna(0.0)

    # Interest expense handling
    if "InterestExpense" in df_wide.columns:
        interest_expense = df_wide["InterestExpense"]
    elif {"EBIT", "PretaxIncome"}.issubset(df_wide.columns):
        interest_expense = (df_wide["EBIT"] - df_wide["PretaxIncome"]).fillna(0.0)
    else:
        interest_expense = 0.0

    # Operating profit
    cost_of_revenue = df_wide["CostOfRevenue"] if "CostOfRevenue" in df_wide.columns else 0.0
    sga = (
        df_wide["SellingGeneralAndAdministration"]
        if "SellingGeneralAndAdministration" in df_wide.columns
        else 0.0
    )

    df_wide["OperatingProfit"] = (
        df_wide["TotalRevenue"]
        - cost_of_revenue
        - sga
        - interest_expense
    )

    # Only keep firms with positive equity
    equity = df_wide["StockholdersEquity"]
    valid = equity > 0

    df_wide.loc[valid, "measure_value"] = (
        df_wide.loc[valid, "OperatingProfit"] / equity.loc[valid]
    )
    df_wide.loc[~valid, "measure_value"] = np.nan

    result = df_wide[["date", "ticker", "measure_value"]].copy()
    result["measure_name"] = "Profitability_OP"
    result = result[["date", "ticker", "measure_name", "measure_value"]]

    return result

def calculate_value(
    accounting_monthly_long: pd.DataFrame,
    market_cap_monthly_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate book-to-market style value factor:

        StockholderEquity / MarketCap

    Inputs
    ------
    accounting_monthly_long:
        date | ticker | measure_name | measure_value

    market_cap_monthly_long:
        date | ticker | measure_name | measure_value
        where measure_name == 'MarketCap'

    Returns
    -------
    date | ticker | measure_name | measure_value
    where measure_name == 'Value_BM'
    """
    acc = accounting_monthly_long.copy()
    mkt = market_cap_monthly_long.copy()

    acc["date"] = pd.to_datetime(acc["date"]) + pd.offsets.MonthEnd(0)
    mkt["date"] = pd.to_datetime(mkt["date"]) + pd.offsets.MonthEnd(0)

    equity = acc.loc[
        acc["measure_name"] == "StockholdersEquity",
        ["date", "ticker", "measure_value"],
    ].copy()
    equity = equity.rename(columns={"measure_value": "StockholdersEquity"})

    market_cap = mkt.loc[
        mkt["measure_name"] == "MarketCap",
        ["date", "ticker", "measure_value"],
    ].copy()
    market_cap = market_cap.rename(columns={"measure_value": "MarketCap"})

    merged = equity.merge(
        market_cap,
        on=["date", "ticker"],
        how="inner",
    )

    merged["measure_value"] = merged["StockholdersEquity"] / merged["MarketCap"]
    merged["measure_name"] = "Value_BM"

    result = merged[["date", "ticker", "measure_name", "measure_value"]].copy()

    return result

def calculate_investment(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 12-month asset growth (Investment_CMA) from long-form data.

    Expected columns:
        date | ticker | measure_name | measure_value
    """
    df = df_long.copy()
    df["date"] = pd.to_datetime(df["date"])

    df_wide = (
        df.pivot_table(
            index=["date", "ticker"],
            columns="measure_name",
            values="measure_value",
            aggfunc="last",
        )
        .reset_index()
        .sort_values(["ticker", "date"])
    )

    if "TotalAssets" not in df_wide.columns:
        raise ValueError("Column 'TotalAssets' not found after pivot.")

    df_wide["Investment_CMA"] = (
        df_wide.groupby("ticker")["TotalAssets"]
        .pct_change(periods=12, fill_method=None)
    )

    # Convert back to long format (only the factor)
    result = df_wide[["date", "ticker", "Investment_CMA"]].copy()
    result = result.rename(columns={"Investment_CMA": "measure_value"})
    result["measure_name"] = "Investment_CMA"

    # Reorder columns to match your schema
    result = result[["date", "ticker", "measure_name", "measure_value"]]

    return result

def calculate_growth(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 12-month business growth as growth in total revenue.

    Expected columns:
        date | ticker | measure_name | measure_value
    """
    df = df_long.copy()
    df["date"] = pd.to_datetime(df["date"])

    df_wide = (
        df.pivot_table(
            index=["date", "ticker"],
            columns="measure_name",
            values="measure_value",
            aggfunc="last",
        )
        .reset_index()
        .sort_values(["ticker", "date"])
    )

    if "TotalRevenue" not in df_wide.columns:
        raise ValueError("Column 'TotalRevenue' not found after pivot.")

    df_wide["Growth_Rev"] = (
        df_wide.groupby("ticker")["TotalRevenue"]
        .pct_change(periods=12, fill_method=None)
    )

    # Convert back to long format (only the factor)
    result = df_wide[["date", "ticker", "Growth_Rev"]].copy()
    result = result.rename(columns={"Growth_Rev": "measure_value"})
    result["measure_name"] = "Growth_Rev"

    # Reorder columns to match your schema
    result = result[["date", "ticker", "measure_name", "measure_value"]]

    return result

def calculate_leverage(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate leverage as TotalDebt / TotalAssets
    and return long format:

        date | ticker | measure_name | measure_value
    """
    required_cols = {"date", "ticker", "measure_name", "measure_value"}
    missing_cols = required_cols - set(df_long.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df = df_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    wide = (
        df.pivot_table(
            index=["date", "ticker"],
            columns="measure_name",
            values="measure_value",
            aggfunc="last",
        )
        .reset_index()
        .sort_values(["ticker", "date"])
    )

    missing_measures = [c for c in ["TotalDebt", "TotalAssets"] if c not in wide.columns]
    if missing_measures:
        raise ValueError(f"Missing required measures after pivot: {missing_measures}")

    wide["measure_value"] = wide["TotalDebt"] / wide["TotalAssets"]

    result = wide[["date", "ticker", "measure_value"]].copy()
    result["measure_name"] = "Leverage_DA"
    result = result[["date", "ticker", "measure_name", "measure_value"]]

    return result

def calculate_momentum(
    close_prices_wide: pd.DataFrame,
    *,
    lookback_months: int = 12,
    skip_months: int = 1,
    measure_name: str = "Momentum_12_1",
    ) -> pd.DataFrame:
    """
    Calculate month-end momentum from wide close price data.

    Expected input format
    ---------------------
    date | ticker_1 | ticker_2 | ...

    where each ticker column contains daily closing prices.

    Default momentum definition
    ---------------------------
    12-1 momentum:
        price(t - skip_months) / price(t - lookback_months) - 1

    With defaults:
        price(t-1) / price(t-12) - 1

    Returns
    -------
    pd.DataFrame
        Long-form dataframe:
        date | ticker | measure_name | measure_value
    """

    if close_prices_wide.index.name == "date" or isinstance(close_prices_wide.index, pd.DatetimeIndex):
        df = close_prices_wide.reset_index()
    else:
        df = close_prices_wide.copy()

    if "date" not in df.columns:
        raise ValueError("Input dataframe must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    ticker_cols = [col for col in df.columns if col != "date"]
    if not ticker_cols:
        raise ValueError("No ticker columns found in input dataframe.")

    # Move to long form first for easier grouped resampling and momentum calculation.
    long_df = df.melt(
        id_vars="date",
        value_vars=ticker_cols,
        var_name="ticker",
        value_name="close",
    )

    long_df["ticker"] = long_df["ticker"].astype(str).str.strip().str.upper()
    long_df = long_df.sort_values(["ticker", "date"])

    # Resample to month-end closes: last observed close in each month for each ticker.
    monthly = (
        long_df
        .groupby(["ticker", pd.Grouper(key="date", freq="ME")], as_index=False)["close"]
        .last()
        .sort_values(["ticker", "date"])
    )

    # Momentum = price(t-skip) / price(t-lookback) - 1
    monthly["price_skip"] = monthly.groupby("ticker")["close"].shift(skip_months)
    monthly["price_lookback"] = monthly.groupby("ticker")["close"].shift(lookback_months)

    monthly["measure_value"] = (
        monthly["price_skip"] / monthly["price_lookback"] - 1.0
    )

    result = monthly[["date", "ticker", "measure_value"]].copy()
    result["measure_name"] = measure_name
    result = result[["date", "ticker", "measure_name", "measure_value"]]

    return result

def download_spx_adjclose(
    start: str = "2010-01-01",
    end: str = "2026-03-31",
    symbol: str = "^GSPC",
    auto_adjust: bool = True,
    ) -> pd.DataFrame:
    """
    Download S&P 500 data from Yahoo and return:

        date | adjClose
    """
    raw = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No data returned for symbol {symbol!r}")

    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    price_series = None

    if isinstance(raw.columns, pd.MultiIndex):
        if auto_adjust:
            close_cols = [col for col in raw.columns if col[0] == "Close"]
            if close_cols:
                price_series = raw.loc[:, close_cols[0]]
        else:
            adj_cols = [col for col in raw.columns if col[0] == "Adj Close"]
            if adj_cols:
                price_series = raw.loc[:, adj_cols[0]]
            else:
                close_cols = [col for col in raw.columns if col[0] == "Close"]
                if close_cols:
                    price_series = raw.loc[:, close_cols[0]]
    else:
        if auto_adjust:
            if "Close" in raw.columns:
                price_series = raw["Close"]
        else:
            if "Adj Close" in raw.columns:
                price_series = raw["Adj Close"]
            elif "Close" in raw.columns:
                price_series = raw["Close"]

    if price_series is None:
        raise ValueError(f"Could not find a usable close column in Yahoo output: {list(raw.columns)}")

    result = (
        price_series.rename("adjClose")
        .reset_index()
        .rename(columns={"Date": "date", "index": "date"})
    )

    result["date"] = pd.to_datetime(result["date"])
    result = result[["date", "adjClose"]].copy()

    return result

def calculate_month_end_beta_and_resid_vol(
    daily_closes: pd.DataFrame,
    spx_adjclose: pd.DataFrame,
    *,
    method: str = "ewma",
    lookback: int = 252,
    ewma_halflife: int = 63,
    min_periods: int | None = None,
    beta_measure_name: str = "beta",
    resid_vol_measure_name: str = "resid_vol",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate daily rolling beta and residual volatility for each stock vs SPX,
    then keep month-end values.

    Parameters
    ----------
    daily_closes
        Wide dataframe of daily stock prices with:
            - DatetimeIndex
            - one column per ticker

    spx_adjclose
        Dataframe with columns:
            - date
            - adjClose

    method
        'equal' or 'ewma'

    lookback
        Rolling lookback window for equal-weight statistics.

    ewma_halflife
        EWMA half-life in trading days.

    min_periods
        Minimum observations before output is produced.
        Defaults to `lookback`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        beta_long, resid_vol_long
    """
    if method not in {"equal", "ewma"}:
        raise ValueError("method must be either 'equal' or 'ewma'")

    if not isinstance(daily_closes.index, pd.DatetimeIndex):
        raise ValueError("daily_closes must have a DatetimeIndex")

    required_spx_cols = {"date", "adjClose"}
    missing_spx_cols = required_spx_cols - set(spx_adjclose.columns)
    if missing_spx_cols:
        raise ValueError(f"spx_adjclose is missing required columns: {sorted(missing_spx_cols)}")

    stocks = daily_closes.copy()
    stocks.index = pd.to_datetime(stocks.index)
    stocks = stocks.sort_index()
    stocks.columns = [str(c).strip().upper() for c in stocks.columns]

    market = spx_adjclose.copy()
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values("date").set_index("date")

    prices = stocks.join(market[["adjClose"]], how="inner")
    if prices.empty:
        raise ValueError("No overlapping dates between daily_closes and spx_adjclose")

    returns = prices.pct_change(fill_method=None)

    stock_returns = returns.drop(columns=["adjClose"])
    market_returns = returns["adjClose"].rename("market_ret")

    aligned = stock_returns.join(market_returns, how="inner").dropna(subset=["market_ret"])
    stock_returns = aligned.drop(columns=["market_ret"])
    market_returns = aligned["market_ret"]

    if min_periods is None:
        min_periods = lookback

    beta_wide = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns, dtype=float)
    resid_vol_wide = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns, dtype=float)

    if method == "equal":
        market_var = market_returns.rolling(window=lookback, min_periods=min_periods).var()

        for ticker in stock_returns.columns:
            stock_series = stock_returns[ticker]

            cov_sm = stock_series.rolling(window=lookback, min_periods=min_periods).cov(market_returns)
            beta_series = cov_sm / market_var
            beta_wide[ticker] = beta_series

            resid = stock_series - beta_series * market_returns
            resid_vol = resid.rolling(window=lookback, min_periods=min_periods).std()

            resid_vol_wide[ticker] = resid_vol

    else:
        market_mean = market_returns.ewm(
            halflife=ewma_halflife,
            adjust=False,
            min_periods=min_periods,
        ).mean()

        market_second = (market_returns ** 2).ewm(
            halflife=ewma_halflife,
            adjust=False,
            min_periods=min_periods,
        ).mean()

        market_var = market_second - market_mean ** 2

        for ticker in stock_returns.columns:
            stock_series = stock_returns[ticker]

            stock_mean = stock_series.ewm(
                halflife=ewma_halflife,
                adjust=False,
                min_periods=min_periods,
            ).mean()

            stock_second = (stock_series ** 2).ewm(
                halflife=ewma_halflife,
                adjust=False,
                min_periods=min_periods,
            ).mean()

            cross_mean = (stock_series * market_returns).ewm(
                halflife=ewma_halflife,
                adjust=False,
                min_periods=min_periods,
            ).mean()

            cov_sm = cross_mean - stock_mean * market_mean
            beta_series = cov_sm / market_var
            beta_wide[ticker] = beta_series

            resid = stock_series - beta_series * market_returns

            resid_mean = resid.ewm(
                halflife=ewma_halflife,
                adjust=False,
                min_periods=min_periods,
            ).mean()

            resid_second = (resid ** 2).ewm(
                halflife=ewma_halflife,
                adjust=False,
                min_periods=min_periods,
            ).mean()

            resid_var = resid_second - resid_mean ** 2
            resid_vol_wide[ticker] = np.sqrt(resid_var)

    beta_monthly = beta_wide.groupby(pd.Grouper(freq="ME")).last()
    resid_vol_monthly = resid_vol_wide.groupby(pd.Grouper(freq="ME")).last()

    beta_long = (
        beta_monthly
        .reset_index()
        .melt(id_vars=beta_monthly.index.name or "date", var_name="ticker", value_name="measure_value")
        .rename(columns={beta_monthly.index.name or "date": "date"})
    )
    beta_long["measure_name"] = beta_measure_name
    beta_long = beta_long[["date", "ticker", "measure_name", "measure_value"]]

    resid_vol_long = (
        resid_vol_monthly
        .reset_index()
        .melt(id_vars=resid_vol_monthly.index.name or "date", var_name="ticker", value_name="measure_value")
        .rename(columns={resid_vol_monthly.index.name or "date": "date"})
    )
    resid_vol_long["measure_name"] = resid_vol_measure_name
    resid_vol_long = resid_vol_long[["date", "ticker", "measure_name", "measure_value"]]

    return beta_long, resid_vol_long    

def calculate_liquidity(
    daily_prices: pd.DataFrame,
    daily_volumes: pd.DataFrame,
    *,
    measure_name: str = "Liquidity",
    monthly_agg: str = "mean",
    scale_factor: float = 1e8
    ) -> pd.DataFrame:
    """
    Calculate a month-end liquidity measure from daily prices and volumes:

        |daily_return| / daily_dollar_volume

    where:
        daily_dollar_volume = daily_price * daily_volume

    Inputs
    ------
    daily_prices
        Wide dataframe of daily prices with:
            - DatetimeIndex
            - one column per ticker

    daily_volumes
        Wide dataframe of daily volumes with:
            - DatetimeIndex
            - one column per ticker

    Parameters
    ----------
    measure_name
        Output factor name. Default: "Liquidity"
    monthly_agg
        Aggregation within month. Supported:
            - "mean"
            - "median"
            - "sum"

    Returns
    -------
    pd.DataFrame
        Long-format dataframe:
            date | ticker | measure_name | measure_value
    """
    if not isinstance(daily_prices.index, pd.DatetimeIndex):
        raise ValueError("daily_prices must have a DatetimeIndex")

    if not isinstance(daily_volumes.index, pd.DatetimeIndex):
        raise ValueError("daily_volumes must have a DatetimeIndex")

    if monthly_agg not in {"mean", "median", "sum"}:
        raise ValueError("monthly_agg must be one of {'mean', 'median', 'sum'}")

    prices = daily_prices.copy()
    volumes = daily_volumes.copy()

    prices.index = pd.to_datetime(prices.index)
    volumes.index = pd.to_datetime(volumes.index)

    prices = prices.sort_index()
    volumes = volumes.sort_index()

    prices.columns = [str(c).strip().upper() for c in prices.columns]
    volumes.columns = [str(c).strip().upper() for c in volumes.columns]

    common_tickers = sorted(set(prices.columns).intersection(volumes.columns))
    if not common_tickers:
        raise ValueError("No overlapping ticker columns between daily_prices and daily_volumes")

    prices = prices[common_tickers]
    volumes = volumes[common_tickers]

    common_dates = prices.index.intersection(volumes.index)
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between daily_prices and daily_volumes")

    prices = prices.loc[common_dates]
    volumes = volumes.loc[common_dates]

    # Daily simple returns
    daily_returns = prices.pct_change(fill_method=None)

    # Daily dollar volume
    daily_dollar_volume = prices * volumes

    # Avoid divide-by-zero
    daily_dollar_volume = daily_dollar_volume.replace(0, np.nan)

    # Daily illiquidity measure
    daily_liquidity = daily_returns.abs() / daily_dollar_volume

    # Aggregate to month-end
    if monthly_agg == "mean":
        monthly_liquidity = daily_liquidity.groupby(pd.Grouper(freq="ME")).mean()
    elif monthly_agg == "median":
        monthly_liquidity = daily_liquidity.groupby(pd.Grouper(freq="ME")).median()
    else:
        monthly_liquidity = daily_liquidity.groupby(pd.Grouper(freq="ME")).sum()

    # Wide -> long
    result = (
        monthly_liquidity
        .reset_index()
        .melt(
            id_vars=monthly_liquidity.index.name or "date",
            var_name="ticker",
            value_name="measure_value",
        )
        .rename(columns={monthly_liquidity.index.name or "date": "date"})
    )

    result["measure_name"] = measure_name

    # scale the measure_value by the scale_factor
    result["measure_value"] = result["measure_value"] * scale_factor
    result = result[["date", "ticker", "measure_name", "measure_value"]]

    return result

if __name__ == "__main__":

    ########################################################
    # Load data
    ########################################################

    print("Loading data...")
    # load accounting data
    # format of the dataframe is as follows:
    # date | ticker | measure_name | measure_value
    accounting_data = pd.read_csv("./accounting_data/accounting_data.csv", parse_dates=["date"])

    # load market cap data
    # format of the dataframe is as follows:
    # date | ticker | measure_name | measure_value
    market_cap = pd.read_csv("./data_files/mkt_cap.csv", parse_dates=["date"])

    # load sector data.
    # This is a dataframe with a column 'ticker' and a column 'sector'
    sectors = pd.read_csv("./data_files/sectors.csv")

    # load the adjusted closing prices
    # the data frame has tickers across the columns and a row of adjusted closing prices for each day
    closing_prices = pd.read_csv("./data_files/daily_closes.csv")

    # load the volume data
    # the data frame has tickers across the columns and a row of trading volume for each day
    volume_data = pd.read_csv("./data_files/daily_volumes.csv")

    # load the unique list of tickers for our universe
    tickers = pd.read_csv("./data_files/tickers.csv")
    tickers = tickers["Ticker"].tolist()

    # clean the daily data
    closing_prices_clean = clean_daily_data(closing_prices, tickers)
    volume_data_clean = clean_daily_data(volume_data, tickers)

    # create monthly accounting data
    # the 3-month lag is to ensure we do not have look-ahead bias
    # on accounting data, which is referenced by period-end dates
    # rather than release dates. 
    # The output is a dataframe with format:
    # date | ticker | measure_name | measure_value
    # where measure_name is one of the following:
    # 'EBITDA', 'TotalRevenue', 'StockholdersEquity', 'TotalAssets', 'TotalDebt', 'PretaxIncome', 'OperatingProfit'
    monthly_accounting_data = prepare_monthly_accounting_data(
                                                        accounting_df=accounting_data,
                                                        lag_months=3,
                                                        monthly_freq="ME"
                                                    )

    # we also need monthly market cap data
    monthly_market_cap = prepare_monthly_market_cap(market_cap)

    ########################################################
    # Build factor data inputs 
    ########################################################
    print("Downloading some s&p data...")
    # 1. market risk and market factors
    spx_adjclose = download_spx_adjclose(
        start="2010-01-01",
        end="2026-03-31",
        symbol="^GSPC",
        auto_adjust=True,
    )

    print("Calculating market risk and market factors...")
    beta_long, resid_vol_long = calculate_month_end_beta_and_resid_vol(
        daily_closes=closing_prices_clean,
        spx_adjclose=spx_adjclose,
        method="equal",          # or "equal"
        lookback=252,
        ewma_halflife=63,
    )

    # 2. Size factor


    # 3. Value factor
    print("Calculating value factor...")
    value_factor = calculate_value(
        accounting_monthly_long=monthly_accounting_data,
        market_cap_monthly_long=monthly_market_cap,
    )

    # 4. Profitability factor
    print("Calculating profitability factor...")
    profitability_factor = calculate_profitability(
        df_long=monthly_accounting_data,
    )

    # 5. Investment factor
    print("Calculating investment factor...")
    investment_factor = calculate_investment(
        df_long=monthly_accounting_data,
    )

    # 6. momentum factor
    print("Calculating momentum factor...")
    momentum_factor = calculate_momentum(closing_prices_clean)
    
    # 7. Growth factor
    print("Calculating growth factor...")
    growth_factor = calculate_growth(monthly_accounting_data)

    # 8. Leverage factor
    print("Calculating leverage factor...")
    leverage_factor = calculate_leverage(monthly_accounting_data)

    # 9. Liquidity factor
    print("Calculating liquidity factor...")
    liquidity_factor = calculate_liquidity(closing_prices_clean, volume_data_clean)

    ########################################################
    # merge the factor data
    ########################################################

    print("Merging factor data...")
    factor_data = pd.concat([beta_long, 
                            resid_vol_long,
                            value_factor, 
                            profitability_factor, 
                            investment_factor, 
                            momentum_factor, 
                            growth_factor, 
                            leverage_factor, 
                            liquidity_factor,
                            monthly_market_cap])

    # remove data before 2012
    factor_data = factor_data[factor_data["date"] >= "2012-01-01"]

    print("Saving factor data...")
    factor_data.to_csv("./factor_data/factor_data.csv", index=False)

    print("Saving daily closes and volumes...")
    closing_prices_clean.to_csv("./data_files/daily_closes_clean.csv", index=True)
    volume_data_clean.to_csv("./data_files/daily_volumes_clean.csv", index=True)
    