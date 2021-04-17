"""
This python program requires two input files (csv format):
1) Combine Dataframe with Core US Fundamental and Bond price by Issuer - Cusip for a security type.
2) Rating file containg rating information at Issuer level, rating infor is available at Bond level Cusip

It returns a data frame combining Core US, EDI bonds & Ratings by fetching most recent rating corresponding to Quarterly date in question.
It Iterates over each Bond Cuisp in size of 5 Bond Cusips.
For each Bond Cuisp --> it loops through all Quarterly Date and find rating from closest rating date corresponding to Quarterly data from Core US dataframe.

To sppeed up 3 nested "for loop", AsyncIO library used to impplement Asyncronous exection of Async Coroutins.

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
# core_cols = ['date_fundamentals','cusip_company_part_fundamentals', 'siccode_fundamentals', 'ticker_fundamentals', 'company_name_fundamentals',   'assets', 'liabilities', 'marketcap', 'ncf',
#        'currentratio', 'receivables', 'payables', 'payoutratio',  'grossmargin', 'ebitdamargin', 'netmargin', 'revenueusd', 'ebitdausd', 'ebitusd', 'debtusd', 'deferredrev', 'pe', 'price', 'inventory',
#        'fcf', 'investments', 'workingcapital', 'capex', 'cashnequsd']

core_cols = ['date_fundamentals','cusip_company_part_fundamentals', 'siccode_fundamentals', 'ticker_fundamentals', 'company_name_fundamentals', 'revenueusd','ebitdausd','ebitdamargin','netinccmnusd',
             'netmargin','cashnequsd','workingcapital','debtusd','liabilities','equityusd','marketcap','ev','fcf','ncfdebt','currentratio','de','divyield','epsusd','payoutratio','evebitda','pb','pe']

core_cols2 = ['date_fundamentals', 'cusip_company_part_fundamentals', 'assets', 'marketcap', 'payables']

# edi_cols = ['date_edi', 'YearMonth_edi', 'mktclosedate', 'maturity_date', 'foreign_key_edi','cusip_company_part_edi', 'uscode', 'isin', 'currency', 'close_edi', 'issuername','coupon', 'security_type']
edi_cols = ['mktclosedate', 'maturity_date', 'cusip_company_part_edi', 'isin', 'uscode', 'secid', 'currency', 'close_edi', 'issuername','coupon', 'sectycd', 'security_type', 'bond_attribute' , 'minimum_denomination' ]
edi_cols2 = ['isin', 'close_edi', 'coupon', 'maturity_date', 'mktclosedate', 'cusip_company_part']

def dateDiff(d1, d2):
       tdelta = (d1 - d2).days
       return tdelta

def getCusip(isin):

       if isin == "":
              return isin

       cusip = isin[2:len(isin)-1]
       return cusip

async def combine_core_rating(core_df, rating_df):
       start_time = time.time()
       combine_df = pd.DataFrame()
       cusip_nums = core_df['cusip'].unique().tolist()
       count = 0

       await asyncio.sleep(0.1)

       print('Strated processing to join Core US cusips with EDI bond price... ')
       for cusip in cusip_nums:
              if count % 10 == 0:
                     print(f'{count} Issuer cusips processed... ')

              cusip_df = core_df.loc[core_df['cusip'] == cusip].sort_values(by = 'date_fundamentals')
              fund_dates = cusip_df['date_fundamentals'].unique().tolist()
              rtg_cusip_df = rating_df.loc[rating_df['cusip'] == cusip].drop_duplicates()
              # isins = rtg_df['cusip'].unique().tolist()

              if rtg_cusip_df.empty:
                     df1 = cusip_df.merge(rtg_cusip_df, how='left', on='cusip')
                     if not combine_df.empty:
                            combine_df = combine_df.append(df1)
                     else:
                            combine_df = df1
                     continue

              rating_dates = rtg_cusip_df['date'].unique().tolist()
              min_rating_date = min(rating_dates)

              try:
                     for date in fund_dates:
                            # if date == datetime.date(2013, 6, 10):
                            #        print(date)
                            if  min_rating_date > date:
                                   continue

                            cusip_df1 = cusip_df.loc[cusip_df['date_fundamentals'] == date]
                            df1 = cusip_df1.merge(rtg_cusip_df, how='inner', on='cusip')
                            df1 = df1.loc[df1['date_fundamentals'] >= df1['date']]

                            if df1.empty:
                                   continue

                            df1['timedelta_2'] = df1.apply(lambda x: dateDiff(x.date_fundamentals, x.date),axis=1)
                            df1 = df1.loc[df1['timedelta_2'] >= 0]
                            df2 = df1.loc[df1['timedelta_2'] == df1['timedelta_2'].min()]

                            if not combine_df.empty:
                                   combine_df = combine_df.append(df2)
                            else:
                                   combine_df = df2
              except Exception as e:
                     print(e)

              count += 1

       end_time = time.time()
       print("--- %s seconds ---" % (end_time - start_time))

       # end_time = time.time()
       # print(f'Total time in seconds: {start_time - end_time}')

       return combine_df

async def main():
       ## Find Most recent Credit rating for a bond at various quarterly date.
       core_bonds_file = 'nt3_cusips_core_df.csv'
       core_bonds_path = Path.joinpath(ROOT_PATH, core_bonds_file)
       core_bond_df = pd.read_csv(core_bonds_path, infer_datetime_format=True, parse_dates=True, )

       date_cols = ['date_fundamentals', 'mktclosedate', 'maturity_date']
       for date_col in date_cols:
              core_bond_df[date_col] = pd.to_datetime(core_bond_df[date_col], errors='coerce').dt.date

       core_bond_df['cusip'] = core_bond_df.apply(lambda x: getCusip(x['isin']), axis=1)

       # Load rating table
       rating_file = 'ranked_ratings.csv'
       rating_path = Path.joinpath(ROOT_PATH, rating_file)
       rating_df = pd.read_csv(rating_path, infer_datetime_format=True, parse_dates=True)
       rating_df['date'] = pd.to_datetime(rating_df['date'], errors='coerce').dt.date

       rating_cols = ['cusip','maturity','rating','date','coupon','issuer_cusip','rank']
       rating__df1 = rating_df[rating_cols].copy()

       core_cusips = sorted(core_bond_df['cusip'].unique().tolist())
       # all_uscode = comp_cusips[0:100]

       chunk_size = 10
       chunks = [core_cusips[x: x + chunk_size] for x in range(0, len(core_cusips), chunk_size)]

       tasks = []
       final_core_df = []

       for chunk in chunks:
              # print(chunk)
              chunk_core_df = core_bond_df.loc[core_bond_df['cusip'].isin(chunk)]
              chunk_edi_df = rating__df1.loc[rating__df1['cusip'].isin(chunk)]

              tasks.append(asyncio.create_task(combine_core_rating(chunk_core_df,chunk_edi_df)))
              # tasks.append(asyncio.ensure_future(get_macd(ticker)))

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
