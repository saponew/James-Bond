"""
This Python takes the dataframe with Core-US data, Bond price data, Ratings and rating change as input.
It classifies the target variable, rating rank change, into two types of categories:

1) Multi class classification - -1 (rating rank decreased), 0 ( rating unchanged), 1 (rating rank increased).
2) Binary classification - 1 (rating rank changed) and  0 ( rating unchanged)

This will return the final_dataframe to be used with other Macro economic data for model building.

"""

import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import time

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

ROOT_PATH = Path("../stock_data/project2/data")

core_cols = ['date_fundamentals', 'cusip_company_part_fundamentals', 'assets', 'marketcap', 'payables']
edi_cols = ['cusip_company_part', 'isin', 'uscode', 'close_edi', 'coupon', 'maturity_date', 'mktclosedate',]


def dateDiff(d1, d2):

    tdelta = (d2 - d1).days/365

    return round(tdelta,3)

def rating_jump(x):
    # print(x)
    change = x['rank_diff']

    if change < 0:
        change = -1
    elif change > 0:
        change = 1

    return change

def rating_jump_2(x):
    # print(x)
    change = x['rank_diff']

    if change < 0 or change > 0:
        change = 1

    return change

def getCusip(isin):
    if isin == "":
        return isin
    cusip = isin[2:len(isin) - 1]
    return cusip

def find_cusip_in_core(core_df, edi_bond_df):
    core_cusips = core_df['cusip_company_part'].unique().tolist()
    edi_bond_cusips = edi_bond_df['cusip_company_part'].unique().tolist()
    common_cusips = list(set(core_cusips) & set(edi_bond_cusips))

    print(common_cusips)
    edi_common_bond_df = None
    if common_cusips:
        edi_common_bond_df = edi_bond_df.loc[edi_bond_df['cusip_company_part'].isin(common_cusips)]

    return edi_common_bond_df

if __name__ == '__main__':

    data_files = ['all_cusips_core_rating_shifted_df.csv', 'nt_cusips_core_rating_shifted_df.csv']
    dfs = []

    for file in data_files:
        file_path = Path.joinpath(ROOT_PATH, file)
        core_bond_df = pd.read_csv(file_path, infer_datetime_format=True, parse_dates=True)

        date_cols = ['date_fundamentals', 'mktclosedate', 'maturity_date', 'date']
        for date_col in date_cols:
            core_bond_df[date_col] = pd.to_datetime(core_bond_df[date_col], errors='coerce').dt.date

        dfs.append(core_bond_df)

    core_us_df = pd.concat(dfs)

    core_us_df['maturity_year'] = core_us_df.apply(lambda x: dateDiff(x.date_fundamentals, x.maturity_date),axis=1)
    core_us_df['rank_change'] = core_us_df.apply(lambda x: rating_jump(x), axis=1)
    core_us_df['rank_change_2'] = core_us_df.apply(lambda x: rating_jump_2(x), axis=1)


    core_us_df.to_csv("C:\stock_data\project2\data\\all_cusips_core_rating_shifted_df_v4.csv", index=False)
    print('Done')

