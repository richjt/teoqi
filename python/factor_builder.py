import pandas as pd
import numpy as np

class FactorBuilder:
    """
    A class for constructing panel factor data, to be used for esimating equity factor models.
    """
    def __init__(self, factor_file: str,
                        sector_file: str = "./factor_data/sectors_clean.csv",
                        ticker_file: str = "./factor_data/ticker_list.csv",
                        winsorise: bool = False,
                        ):
        
        
        # read in the factor data
        self.factor_data = pd.read_csv(factor_file, parse_dates=["date"])

        # read in the sector file
        sector = pd.read_csv("sectors_clean.csv")
        self.sector_map = dict(zip(sector["ticker"], sector["sector"]))

        # read in the ticker list
        t = pd.read_csv("./factor_data/ticker_list.csv")
        self.tickers = t["ticker"].to_list()

        # whether or not the factors should be winsorised before z-scoring 
        self.winsorise = winsorise

        # infer the factor names from the factor data
        self.factor_names = self.factor_data["measure_name"].unique()

    def calculate_factor_panel(self, 
                                factor_name: str,
                                request_date: pd.Timestamp,
                                calc_median_by_sector: bool = True) -> pd.DataFrame:
        """
        Retrieves the data for the specified factor name and request date and creates a z-scored vector for that factor

        If calc_median_by_sector is True, any missing data is filled by the median value within its sector. And then a final sweep is done
        to replace any missing data with the global median (in case any sectors have little or no data).

        If calc_median_by_sector is False, only the global median sweep is done. 

        The class level setting for winsorising determines if the factor data should be winsorised before z-scores are calculated.
        """

        # get the factor data for the given date
        data = self.factor_data[self.factor_data["date"] == request_date.normalize()]
        data = self.factor_data[self.factor_data["ticker"].isin(self.tickers)]
        data = data[data["measure_name"] == factor_name]
        
        wide = data.pivot_table(
            index="ticker",
            columns="measure_name",
            values="measure_value",
            aggfunc="first"  # or "mean", "max", etc.
        )

        wide = wide.reindex(self.tickers)
        wide["sector"] = wide.index.map(self.sector_map)

        nan_idx = wide.index[wide[factor_name].isna()]
        wide.loc[nan_idx, :]

        if len(nan_idx) > 0:
            wide[factor_name] = wide[factor_name].fillna(
                wide.groupby("sector")[factor_name].transform("median"))
            wide[factor_name] = wide[factor_name].fillna(wide[factor_name].median())

        if self.winsorise:
            lower_q = wide[factor_name].quantile(0.01)
            upper_q = wide[factor_name].quantile(0.99)
            wide[factor_name] = wide[factor_name].clip(lower=lower_q, upper=upper_q)
        
        wide[factor_name] = (wide[factor_name] - wide[factor_name].mean()) / wide[factor_name].std()
        wide = wide[factor_name]
        return wide
        
    def build_cross_sectional_panel(self,
                                    request_date: pd.Timestamp,
                                    calc_median_by_sector: bool = True) -> pd.DataFrame:

        """
        For each factor in the input factor data for the given date, a z-scored factor panel is created plus
        sector loadings (1,0). 
        """
        out_parts = []
        for factor_name in self.factor_names:
            out_parts.append(self.calculate_factor_panel(factor_name, request_date, calc_median_by_sector))
        
        op_df = pd.concat(out_parts, axis=1)
        op_df = self.add_sector_columns(op_df)
        return op_df

    def add_sector_columns(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        # for each sector we add a column that has a 1 if the ticker is in that sector and 0 otherwise
        sector_cols = pd.DataFrame(index=factor_df.index)
        for sector in self.sector_map.values():
            sector_cols[sector] = factor_df.index.map(lambda x: 1 if self.sector_map[x] == sector else 0)
        return pd.concat([factor_df, sector_cols], axis=1)
    


