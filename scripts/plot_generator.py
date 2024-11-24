import taxiplot
from sqlalchemy import create_engine
import os
import time

import plotly.io as pio
from PIL import Image

import boto3


os.environ['TZ'] = 'Europe/Helsinki'
time.tzset()

def main():
    print("starting script")
    sql_engine = create_engine('postgresql://*******.eu-north-1.rds.amazonaws.com:5432/') # hidden

    t = taxiplot.time_now_15()
    preds_df = taxiplot.get_sql_table("preds", sql_engine)
    preds_df = preds_df[preds_df["datetime"]>=t]
    preds = preds_df["y"].values
    rides_df = taxiplot.get_ride_data(sql_engine = sql_engine)

    fig = taxiplot.print_forecast(preds, rides_df, t, sql_engine=sql_engine, ME=False)

    im = pio.write_image(fig, "plot_sum.png", width=6*200, height=2.5*200, scale=3)

    s3 = boto3.client('s3')
    s3.upload_file("plot_sum.png", "taxipoint-pp", "plot_sum.png")

    # # #

    t = taxiplot.time_now_15()
    preds_df = taxiplot.get_sql_table("preds_me", sql_engine)
    preds_df = preds_df[preds_df["datetime"]>=t]
    preds = preds_df["y"].values
    rides_df = taxiplot.get_ride_data(sql_engine = sql_engine, ME=True)

    fig = taxiplot.print_forecast(preds, rides_df, t, sql_engine=sql_engine, ME=True)

    im = pio.write_image(fig, "plot_sum_me.png", width=6*200, height=2.5*200, scale=3)

    s3 = boto3.client('s3')
    s3.upload_file("plot_sum_me.png", "taxipoint-pp", "plot_sum_me.png")

    print("new plots saved!")
    del sql_engine, t, preds_df, rides_df, fig, im, s3

