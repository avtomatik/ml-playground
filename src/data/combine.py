#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 22:02:35 2023

@author: alexandermikhailov
"""


from operator import itemgetter

import duckdb
import pandas as pd


def combine_cobb_douglas(series_number: int = 3) -> pd.DataFrame:

    SERIES_IDS_EXT = {
        "CDT2S4": ("dataset_usa_cobb-douglas", "capital"),
        "CDT3S1": ("dataset_usa_cobb-douglas", "labor"),
        "J0014": ("dataset_uscb", "product"),
        "J0013": ("dataset_uscb", "product_nber"),
        "DT24AS01": ("dataset_douglas", "product_rev"),
    }

    selected = list(SERIES_IDS_EXT.keys())

    con = duckdb.connect()

    df = con.execute(
        f"""
        SELECT year, series_code, value
        FROM 'data/bronze/*.parquet'
        WHERE series_code IN ({",".join([f"'{s}'" for s in selected])})
    """
    ).df()

    wide = (
        df.pivot(index="year", columns="series_code", values="value")
        .rename(
            columns=dict(
                zip(
                    SERIES_IDS_EXT.keys(),
                    map(itemgetter(1), SERIES_IDS_EXT.values()),
                )
            )
        )
        .iloc[:, :series_number]
        .dropna()
        .sort_index()
    )

    return wide
