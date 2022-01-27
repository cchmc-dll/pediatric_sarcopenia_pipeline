import os
from collections import namedtuple

import pandas as pd
import csv
import numpy as np
import scipy.stats as stats


POUNDS_TO_KG = 2.20462262185
INCHES_TO_CM = 2.54

CDCDataEntry = namedtuple('CDCDataEntry', ['Sex', 'factor', 'L', 'M', 'S'])


def build_cdc_stats_table(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader, None)
        table = {}
        entries = []
        for row in csv_reader:
            entry = CDCDataEntry(
                Sex=row[0],
                factor=row[1],
                L=float(row[2]),
                M=float(row[3]),
                S=float(row[4])
            )
            entries.append(entry)

        # min factors represent minimum numbers, handled differently than others unfortunately
        min_factor = float(entries[0].factor)
        min_factors_removed = filter(lambda r: r.factor != min_factor, entries)

        for entry in min_factors_removed:
            table[(int(entry.Sex), np.floor(float(entry.factor)))] = entry

        return CDCStatsTable(table, min_factor)



class CDCStatsTable:
    def __init__(self, table, min_factor):
        self.table = table
        self.min_factor = min_factor

    def variables_for(self, sex, factor):
        if factor == self.min_factor:
            return self.table[0]
        else: 
            return self.table[(sex, np.floor(float(factor)))]


def get_cdc_dataframe(path, sheet_name):
    return pd.ExcelFile(path).parse(sheet_name)


def get_nsqip_dataframe(path):
    nsqip_xl = pd.ExcelFile(nsqip)
    nsqip_df = nsqip_xl.parse('Sheet1')
    return nsqip_df


def age_in_days_to_months(age_in_days):
    return age_in_days / (365.25/12)

def age_in_years_to_months(age_in_years):
    return age_in_years * 12

def add_string_male_female(df):
    df['Sex_str'] = df.Sex.apply(lambda s: 'Male' if s == 1 else 'Female')


def variables_for_age_in_months_and_sex(cdc_df, age_in_mos, sex):
    if age_in_mos == 0:
        return cdc_df.iloc[0]
    else:
        age_less_than = cdc_df[(np.floor(cdc_df.Agemos) <= np.floor(age_in_mos)) & (cdc_df.Sex_str == sex)]
        return age_less_than.iloc[-1]


def zscore_for_age_in_months_and_sex(cdc_df, measurement, age_in_mos, sex):
    v = variables_for_age_in_months_and_sex(cdc_df, age_in_mos, sex)
    return zscore_for_measurement(measurement, v['L'], v['M'], v['S'])


def zscore_for_measurement(measurement, L, M, S):
    return ((measurement/M)**(L) - 1) / (L*S)
    

def percentile_for_zscore(zscore):
    return stats.norm.cdf(zscore)


def calculate_zscore_for_weight(nsqip_df, cdc_df):
    def zscore_per_row(row):
        if row['WEIGHT'] < 0:
            return np.nan
        else:
            return zscore_for_age_in_months_and_sex(
                cdc_df,
                row['WEIGHT'] / POUNDS_TO_KG,
                age_in_days_to_months(row['AGE_DAYS']),
                row['SEX']
            )

    return nsqip_df.apply(zscore_per_row, axis=1)


def calculate_zscore_for_height(nsqip_df, cdc_df):
    def zscore_per_row(row):
        if row['HEIGHT'] < 0:
            return np.nan
        else:
            return zscore_for_age_in_months_and_sex(
                cdc_df,
                row['HEIGHT'] * INCHES_TO_CM,
                age_in_days_to_months(row['AGE_DAYS']),
                row['SEX']
            )

    return nsqip_df.apply(zscore_per_row, axis=1)


# nsqip = '~/research/PNSQIP_CPT_abbreviated.xlsx'
# wtage = os.path.expanduser('./wtagecombined.xlsx')
# lnage = os.path.expanduser('./lengthstaturecombinedat24_5months.xlsx')
#
# nsqip_xl = pd.ExcelFile(nsqip)
# nsqip_df = nsqip_xl.parse('Sheet1')
#
# wtage_df = get_cdc_dataframe(wtage, 'Sheet1')
# lnage_df = get_cdc_dataframe(lnage, 'Sheet1')
# add_string_male_female(wtage_df)
# add_string_male_female(lnage_df)
#
# print("Calculating Weight Percentiles...")
# nsqip_df['wt_zscore'] = calculate_zscore_for_weight(nsqip_df, wtage_df)
#
# print("Calculating Height Percentiles...")
# nsqip_df['ht_zscore'] = calculate_zscore_for_height(nsqip_df, lnage_df)
