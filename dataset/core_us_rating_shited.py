"""
This Python takes the dataframe with Core-US data, Bond price data and Ratings and
fetches the 12 month forward rating corresponding to quarterly date in question

It shift the 12 months future ratings to current quarterly record. it also calculate change in rating rank and insert as new column - rank_diff.
This will create the data frame all bonds in scope for model building.

"""

import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import time
import asyncio
import os

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

ROOT_PATH = Path("../stock_data/project2/data")

def dateDiff(d1, d2):
       tdelta = (d1 - d2).days
       return tdelta

def getCusip(isin):
       if isin == "":
              return isin

       cusip = isin[2:len(isin)-1]
       return cusip

async def get_shfit_rating(core_df):
       start_time = time.time()
       shifted_df = pd.DataFrame()
       cusip_nums = core_df['cusip'].unique().tolist()
       count = 0

       await asyncio.sleep(0.1)

       # test_cols = ['cusip', 'date_fundamentals', 'date_fundamentals_shifted', 'rating', 'rating_shifted' 'rank','rank_shifted']
       print('Strated processing Core US cusips to shift ratings... ')

       for cusip in cusip_nums:
              try:
                     cusip_df = core_df.loc[core_df['cusip'] == cusip].sort_values(by='date_fundamentals')
                     cusip_df['date_fundamentals_shifted'] = cusip_df['date_fundamentals'].shift(-2)
                     cusip_df['rating_shifted'] = cusip_df['rating'].shift(-2)
                     cusip_df['rank_shifted'] = cusip_df['rank'].shift(-2)
                     cusip_df['rank_diff'] = cusip_df['rank'] - cusip_df['rank_shifted']

                     if not shifted_df.empty:
                            shifted_df = shifted_df.append(cusip_df)
                     else:
                            shifted_df = cusip_df

              except Exception as e:
                     print(e)

              count += 1

       if count % 10 == 0:
              print(f'{count} cusips processed... ')

       end_time = time.time()
       print("--- %s seconds ---" % (end_time - start_time))

       return shifted_df

async def main():
       data_files = ['nt3_cusips_core_rating_df.csv', 'nt4_cusips_core_df_rating.csv']
       dfs = []

       for file in data_files:
              file_path = Path.joinpath(ROOT_PATH, file)
              core_bond_df = pd.read_csv(file_path, infer_datetime_format=True, parse_dates=True)

              date_cols = ['date_fundamentals', 'mktclosedate', 'maturity_date', 'date']
              for date_col in date_cols:
                     core_bond_df[date_col] = pd.to_datetime(core_bond_df[date_col], errors='coerce').dt.date

              dfs.append(core_bond_df)

       core_us_df = pd.concat(dfs)

       core_cusips = sorted(core_us_df['cusip'].unique().tolist())
       # all_uscode = comp_cusips[0:100]

       chunk_size = 20
       chunks = [core_cusips[x: x + chunk_size] for x in range(0, len(core_cusips), chunk_size)]

       tasks = []
       final_core_df = []
       for chunk in chunks:
              # print(chunk)
              chunk_core_df = core_us_df.loc[core_us_df['cusip'].isin(chunk)]
              tasks.append(asyncio.create_task(get_shfit_rating(chunk_core_df)))

       await asyncio.gather(*tasks)

       for task in tasks:
              if len(task.result()) == 0:
                     continue
              final_core_df.append(task.result())

       return final_core_df

if __name__ == '__main__':

       start_time = time.time()
       res = asyncio.run(main())

       final_df = pd.concat(res)

       end_time = time.time()
       print("--- %s seconds ---" % (end_time - start_time))
