import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import time
import asyncio
import os

"""
This python program requires twp input files (csv format): 
1) Core US Fundamental Quarterly Data by Issuer Cusip 
2) EDI_Bond data for one of the security type from  - Bond, Debentures, Medium Term Note, Note
It returns two data_frames Core US abd EDI bonds with Issuer Cusips Common in both the input Data set 
"""

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

ROOT_PATH = Path("../stock_data/project2/data")

core_cols = ['date_fundamentals','cusip_company_part_fundamentals', 'siccode_fundamentals', 'ticker_fundamentals', 'company_name_fundamentals', 'revenueusd','ebitdausd','ebitdamargin','netinccmnusd',
             'netmargin','cashnequsd','workingcapital','debtusd','liabilities','equityusd','marketcap','ev','fcf','ncfdebt','currentratio','de','divyield','epsusd','payoutratio','evebitda','pb','pe']

edi_cols = ['mktclosedate', 'maturity_date', 'cusip_company_part_edi', 'isin', 'uscode', 'secid', 'currency', 'close_edi', 'issuername','coupon', 'sectycd', 'security_type', 'bond_attribute' , 'minimum_denomination']
edi_cols2 = ['isin', 'close_edi', 'coupon', 'maturity_date', 'mktclosedate', 'cusip_company_part']


def main():
       core_us_file = 'core_us_fundamentals.csv'
       core_path = Path.joinpath(ROOT_PATH, core_us_file)
       # Load Core US Fundamental table
       df = pd.read_csv(core_path, usecols=core_cols, infer_datetime_format=True, parse_dates=True,)

       # Subset of Core US df
       core_df = df[core_cols]
       core_df['date_fundamentals'] = pd.to_datetime(df['date_fundamentals']).dt.date
       core_df_test = core_df[core_cols].copy()
       core_df_test = core_df_test.rename(columns={'cusip_company_part_fundamentals': 'cusip_company_part'})

       # EDI Bond Price Table
       edi_bond_file = 'edi_bond_price_NT_4.csv'
       edi_path = Path.joinpath(ROOT_PATH, edi_bond_file)
       df_edi_bond_price = pd.read_csv(edi_path, infer_datetime_format=True, parse_dates=True)
       df_edi_bond_price = df_edi_bond_price.rename(columns={'cusip_company_part_edi': 'cusip_company_part'})

       # Convert string into Datetime.Date
       edi_dates = ['mktclosedate', 'maturity_date']
       for date in edi_dates:
              df_edi_bond_price[date] = pd.to_datetime(df_edi_bond_price[date], errors='coerce').dt.date

       bond_cusip = df_edi_bond_price['cusip_company_part'].unique().tolist()
       core_bond_df = core_df_test.loc[core_df_test['cusip_company_part'].isin(bond_cusip)].sort_values(by=['cusip_company_part', 'date_fundamentals'])

       # Issuer Cusips avilable in EDI Bond Price table
       comp_cusips = core_bond_df['cusip_company_part'].unique().tolist()

       # Bond price Dataframe from EDI bond price only for the Cusip available in Core US Fundamental
       edi_cusip_df = df_edi_bond_price.loc[df_edi_bond_price['cusip_company_part'].isin(comp_cusips)]

       return core_bond_df, edi_cusip_df

if __name__ == '__main__':

       core_df, edi_bond_df = main()

       print("Done!")
