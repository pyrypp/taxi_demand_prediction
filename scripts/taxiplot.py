# standard
import datetime as dt

# sql
from sqlalchemy import create_engine
from sqlalchemy.types import *
import psycopg2

# data analysis
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import argrelmin

# data viz
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def save_to_sql_feedback(arvosana, teksti, sql_engine):
    dt_ = dt.datetime.now()
    df = pd.DataFrame({"datetime":[dt_],"arvosana":[arvosana], "teksti":[teksti]})
    df.to_sql("feedback",sql_engine, if_exists='append', index=False, dtype={"arvosana":Integer, "teksti":VARCHAR(255)})

def time_now_15():
    return pd.Timestamp(dt.datetime.now()).floor("15min")

def get_sql_table(table, sql_engine):
    conn = sql_engine.connect()

    if table == "rides":
        t_now = time_now_15() - dt.timedelta(days=1)
        date_f = str(t_now.date())
        df = pd.read_sql(f"SELECT * FROM {table} WHERE date > '{date_f}'", con=conn)

    df = pd.read_sql(f"SELECT * FROM {table}", con=conn)


    try:
        df = df.sort_values(by=["date", "time"], ascending=True)
    except:
        None
    conn.close()
    return df

def get_ride_data(x_time=time_now_15(), grouper_freq="15min", savgol=9, sql_engine=None, ME=False):
    """
    returns the past 96 values
    """

    if ME:
        savgol=5

    x_time_dt = pd.to_datetime(x_time)

    # time
    start_dt = x_time - dt.timedelta(days = 1) - dt.timedelta(minutes = 15)
    end_dt = x_time - dt.timedelta(minutes = 15)

    # get table
    rides_df = get_sql_table("rides", sql_engine)

    # create "datetime" and "sum", group
    rides_df["datetime"] = pd.to_datetime(rides_df['date'].astype(str) + ' ' + rides_df['time'].astype(str), format='%Y-%m-%d %H:%M:%S')
    rides_df = rides_df.groupby(pd.Grouper(key="datetime", freq=grouper_freq))[["FT", "TH", "ME", "MU"]].sum().reset_index()
    rides_df["sum"] = rides_df["FT"] + rides_df["TH"] + rides_df["ME"] + rides_df["MU"]

    # ME
    if ME:
        rides_df = rides_df.drop(columns=["sum"])
        rides_df["sum"] = rides_df["ME"].copy()


    # time and date features
    rides_df["time"] = rides_df["datetime"].dt.time
    rides_df["date"] = rides_df["datetime"].dt.date

    # dt filter
    rides_df = rides_df[rides_df["datetime"]>=start_dt]
    rides_df = rides_df[rides_df["datetime"]<=end_dt]

    # outliers
    rides_df[["FT", "TH", "ME", "MU"]] = rides_df[["FT", "TH", "ME", "MU"]].applymap(lambda v: np.nan if v > 60 else v)
    rides_df = rides_df.ffill()

    # ensure no missing data at end
    rides_df.loc[len(rides_df.index)] = [end_dt, 0, 0, 0, 0, 0, end_dt.time(), end_dt.date()]
    rides_df = rides_df.drop_duplicates("datetime")

    if savgol > 0:
        rides_df["sum"] = savgol_filter(rides_df["sum"], savgol, 2)


    rides_df["sum"] = rides_df["sum"].clip(lower=0)

    rides_df.reset_index(inplace=True, drop=True)
    
    return rides_df

###

def trans_col(string):
    r = int(string[5:].split(",")[0])
    g = int(string[5:].split(",")[1])
    b = int(string[5:].split(",")[2])

    r_new = np.mean([r, 255])
    g_new = np.mean([g, 255])
    b_new = np.mean([b, 255])

    return f"rgba({r_new},{g_new},{b_new},0.5)"

top_colors = [
    "rgba(0,191,255,  0.8)",
    "rgba(255,179,102,  0.7)",
    "rgba(255,80,0,  0.9)",
    "rgba(256,0,0,  0.8)"
]

colorscales = [[(0, "rgba(256,256,256,  0)"), (0.4, "rgba(256,256,256,  0)"), (0.8, trans_col(top_colors[0])), (1, top_colors[0])],
               [(0, "rgba(256,256,256,  0)"), (0.4, "rgba(256,256,256,  0)"), (0.8, trans_col(top_colors[1])), (1, top_colors[1])],
               [(0, "rgba(256,256,256,  0)"), (0.4, "rgba(256,256,256,  0)"), (0.8, trans_col(top_colors[2])), (1, top_colors[2])],
               [(0, "rgba(256,256,256,  0)"), (0.4, "rgba(256,256,256,  0)"), (0.8, trans_col(top_colors[3])), (1, top_colors[3])] 
]

def select_color(value, ME):
    if ME:
        x = 0.5
    else:
        x = 1

    if value < 180 * x:
        return 0
    elif 180 * x <= value < 330 * x:
        return 1
    elif 330 * x <= value < 490 * x:
        return 2
    elif value >= 490 * x:
        return 3
    else:
        return None

def round_to_next_series_hour(datetime_value):
    hour = datetime_value.hour
    
    next_hour = ((hour // 3) + 1) * 3
    
    if next_hour >= 24:
        next_hour = 0
        datetime_value += pd.Timedelta(days=1)
    
    rounded_datetime = datetime_value.replace(hour=next_hour)
    
    return rounded_datetime


def print_forecast(preds, rides_df_, t, sql_engine, ME):
    dt_range = pd.date_range(start=t, periods=len(preds), freq="15min")


    df = pd.DataFrame(columns=["dt", "y"])
    df["dt"] = dt_range
    df["y"] = preds

    df["y"] = df["y"].clip(lower=0)

    minima = argrelmin(df.y.values, order=2)[0]
    df["group"] = False
    df.loc[minima, "group"] = True

    df["group"] = df["group"].replace({False:np.nan})
    df["group"] = df["group"].cumsum()

    df = df.ffill()
    df = df.bfill()

    df["group"] = df["group"].astype(str)

    df["y2"] = df["y"].copy()

    cutoff = df.y.quantile(0.5)

    df.loc[df.y.values<cutoff, "y2"] = np.nan
    df_gp = df.groupby("group")["y2"].sum().reset_index().rename(columns={"y2":"sum"})
    df = df.merge(df_gp, on="group", how="left")
    df = df.drop(columns="y2")

    if ME:
        df["group"] = df["group"].where(df["sum"]>20, np.nan)
    else:
        df["group"] = df["group"].where(df["sum"]>150, np.nan)



    df["group"] = df["group"].fillna(np.nan)

    df["sum"] = df["sum"].where(~df.group.isna(), np.nan)

    df = df.bfill()
    df = df.ffill()
    df = df.fillna(0)

    if ME:
        df["sum"] = df["sum"].round(1).astype(int)
    else:
        df["sum"] = df["sum"].round(-1).astype(int)


    dfs = []
    for group in df["group"].unique():
        df2 = df[df["group"]==group]

        next_row_index = df2.index[-1] + 1
        if next_row_index < len(df):
            next_row = df.iloc[next_row_index:next_row_index + 1]
            df2 = pd.concat([df2, next_row])

        df2 = df2.reset_index(drop=True)

        df2["max"] = df2["y"].max()
        df2["dt_max"] = df2[df2["y"]==df2["max"][0]]["dt"]

        dfs.append(df2)



    # # # # # #     # # # # # #     # # # # # #
    # # # # # #     # # # # # #     # # # # # #
    # # # # # #     # # # # # #     # # # # # #


    rides_df_ = get_ride_data(t, savgol=0, sql_engine=sql_engine, ME=ME)
    rides_df_ = rides_df_[["datetime", "sum"]]

    all_dfs = pd.concat(dfs)

    all_dt = all_dfs["dt"]
    all_dt = pd.concat([rides_df_["datetime"], all_dt])

    all_dfs = all_dfs[["dt","y"]].rename(columns={"dt": "datetime", "y":"sum"})

    # rides_df_.loc[len(rides_df_)] = [dfs[0]["dt"][0], preds[0]]
    rides_df_ = pd.concat([rides_df_, all_dfs[["datetime", "sum"]][:9]])
    rides_df_["sum"] = savgol_filter(rides_df_["sum"], 9, 2)
    rides_df_ = rides_df_[:-9]

    rides_df_.loc[len(rides_df_.index)] = [t, dfs[0]["y"][0]]

    fig = make_subplots()

    for df_to_plot in dfs:
        color_level = select_color(df_to_plot["sum"][0], ME)
        fig.add_trace(go.Scatter(
            x=df_to_plot["dt"], 
            y=df_to_plot["y"], 
            fill="tozeroy", 
            mode="lines", 
            line=dict(color="#888888"),
            fillgradient=dict(type="vertical", colorscale=colorscales[color_level]),
            line_shape="spline"
        ))
        if df_to_plot["dt_max"].max() != df_to_plot["dt"][0] and df_to_plot["dt_max"].max() != df_to_plot["dt"].values[-1]:
            if ME:
                delta = 0.5
            else:
                delta = 2
            fig.add_annotation(x=df_to_plot["dt_max"].max(), y=df_to_plot["y"].max()+delta, text=str(df_to_plot["sum"][0]), showarrow=False, font=dict(size=16))



    fig.add_trace(go.Scatter(x=rides_df_["datetime"], y=rides_df_["sum"], name="y_real", marker=dict(color="#555555")))

    # Generate custom tickvals and ticktext
    tickvals = []
    ticktext = []

    # Set interval for time ticks (every 3 hours in this example)
    time_interval = 2
    " times"
    for i, dt_ in enumerate(all_dt):
        if dt_.hour % time_interval == 0 and dt_.minute == 0:
            tickvals.append(dt_)
            ticktext.append(dt_.strftime('%H:%M'))

    # dates
    for i,dt_ in enumerate(tickvals):
        if dt_.time() == dt.time(0):     
            ticktext[i] = ticktext[i] + f"<br>{dt_.day}.{dt_.month}"

    # first date
    if len(ticktext[0]) <= 5:
        ticktext[0] = ticktext[0] + f"<br>{tickvals[0].day}.{tickvals[0].month}"

    fig.update_layout(
                    # width=1800, 
                    # height=600,
                    template="simple_white",
                    yaxis=dict(dtick=10, fixedrange=True),
                    xaxis=dict(fixedrange=True),
                    showlegend=False,
                    hovermode=False,
                    yaxis_title="Asiakkaita / 15 min",
                    # font=dict(size=16)
                    )

    if ME:
        fig.update_layout(
                    yaxis=dict(dtick=2)
                    )

    fig.update_xaxes(
        tickvals=tickvals,
        ticktext=ticktext
    )

    delta 

    y_range_max = max(all_dfs["sum"].max(), rides_df_["sum"].max()) + 5
    fig.update_yaxes(range=[0,y_range_max])

    fig.add_vline(x=t, line_width=2, line_color="black", opacity=1)


    # t_str = str(dt.datetime.now().strftime("%H:%M"))
    t_now = dt.datetime.now()
    t_str = str(t_now.strftime("%H:%M"))

    shift = 0.03

    fig.add_annotation(x=t, y=1.091+shift, text=f"{t_str}", showarrow=False, yref="paper", font=dict(size=26))

    fig.add_annotation(x=t-pd.Timedelta(hours=6), y=1.08+shift, text="havainto", showarrow=False, yref="paper", font=dict(size=20))
    fig.add_annotation(x=t+pd.Timedelta(hours=6), y=1.08+shift, text="ennuste", showarrow=False, yref="paper", font=dict(size=20))


    fig.update_xaxes(showgrid=True, gridwidth=2)
    # fig.update_yaxes(showgrid=True)

    return fig

# print_forecast(preds, rides_df, t)