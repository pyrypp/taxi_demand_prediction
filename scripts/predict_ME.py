# standard
import os
import time
import json
import datetime as dt
import warnings
import gc


# data scraping
import requests
import xml.etree.ElementTree as ET

# data analysis
import pandas as pd
import numpy as np

# sql
from sqlalchemy import create_engine
from sqlalchemy.types import *
import psycopg2

# modeling
import tensorflow as tf
import joblib

from scipy.signal import savgol_filter

cwd = ""

# linux
os.environ['TZ'] = 'Europe/Helsinki'
time.tzset()
cwd = "/home/ubuntu/"


# initializing sys
pd.options.mode.copy_on_write = True
warnings.simplefilter(action='ignore', category=FutureWarning)

# initializing SQL
sql_engine = create_engine('postgresql://******.eu-north-1.rds.amazonaws.com:5432/') # hidden

# Finavia
url_finavia      = 'https://api.finavia.fi/flights/public/v0/flights/arr/HEL'
headers_finavia  = {"Accept":"application/xml","app_id": "******", "app_key":"******"} # hidden


def get_sql_table(table):
    with sql_engine.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", con=conn)
        try:
            df = df.sort_values(by=["date", "time"], ascending=True)
        except:
            None
    return df

def time_now_15():
    return pd.Timestamp(dt.datetime.now()).floor("15min")

def convert_to_float(x):
    try:
        return float(x)
    except:
        return None

def get_finvia_data_realtime():
    # Get data
    response = requests.get(url_finavia, headers=headers_finavia)
    xml_data = response.text

    # Parse the XML data
    root = ET.fromstring(xml_data)

    # Initialize empty lists to store data
    fltnr_list = []
    sdt_list = []
    actype_list = []

    # Extract data from XML and append to lists
    for flight in root.findall('.//{http://www.finavia.fi/FlightsService.xsd}flight'):
        fltnr_list.append(flight.find('{http://www.finavia.fi/FlightsService.xsd}fltnr').text)
        sdt_list.append(flight.find('{http://www.finavia.fi/FlightsService.xsd}sdt').text)
        actype_list.append(flight.find('{http://www.finavia.fi/FlightsService.xsd}actype').text)

    data = {'fltnr': fltnr_list, 'sdt': sdt_list, 'actype': actype_list}
    flights_df = pd.DataFrame(data)

    # Dataframe manipulations
    flights_df["datetime"] = pd.to_datetime(flights_df["sdt"])

    # flights_df["datetime"] = flights_df["datetime"].dt.tz_localize(pytz.utc)
    flights_df["datetime"] = flights_df["datetime"].dt.tz_convert("Europe/Helsinki")
    flights_df["datetime"] = flights_df["datetime"].dt.tz_localize(None)

    # flights_df["datetime"] = pd.to_datetime(flights_df["datetime"])

    # flights_df["date"] = flights_df["datetime"].dt.date.astype(str)
    # flights_df["time"] = flights_df["datetime"].dt.time.astype(str)
    # flights_df = flights_df[["date", "time", "fltnr", "actype"]]
    flights_df = flights_df[["datetime", "fltnr"]]

    flights_df = flights_df.reset_index(drop=True)

    return flights_df

###

def get_ride_data(x_time=time_now_15(), grouper_freq="15min", savgol=9, ME=False):
    """
    returns the past 96 values
    """
    x_time_dt = pd.to_datetime(x_time)

    # time
    start_dt = x_time - dt.timedelta(days = 1) - dt.timedelta(minutes = 15)
    end_dt = x_time - dt.timedelta(minutes = 15)

    # get table
    rides_df = get_sql_table("rides")

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

def get_flight_data(x_time=time_now_15(), grouper_freq="15min"):
    """
    returns next 96 values
    USE X_TIME ONLY FOR VALUES OVER 24H OLD!!
    """

    # time
    start_dt = x_time
    end_dt = x_time + dt.timedelta(days = 1)

    # get flt_num cols
    with open(f"{cwd}flt_nums.json","r") as f:
        flt_nums = json.load(f)

    flt_nums = ["datetime"] + flt_nums

    # get table
    if x_time == time_now_15():
        flights_df = get_finvia_data_realtime()
    else:
        flights_df = get_sql_table("flights")
        flights_df["datetime"] = pd.to_datetime(flights_df['date'].astype(str) + ' ' + flights_df['time'].astype(str), format='%Y-%m-%d %H:%M:%S')
        flights_df = flights_df[["datetime", "fltnr"]]


    # # SHIFT !!
    flights_df["datetime"] = flights_df["datetime"] + dt.timedelta(hours=1)

    # dt filter
    flights_df = flights_df[flights_df["datetime"]>=start_dt]
    flights_df = flights_df[flights_df["datetime"]<end_dt]
    
    # flight number df (wide_df)
    flight_nr_df = flights_df.groupby("datetime")["fltnr"].apply(list).reset_index()
    flight_nr_df = flight_nr_df.explode(column="fltnr")
    flight_nr_df = pd.get_dummies(flight_nr_df, columns=["fltnr"], dtype="int")
    flight_nr_df = flight_nr_df.groupby("datetime").apply("max").reset_index()
    flight_nr_df = flight_nr_df.groupby(pd.Grouper(key="datetime", freq=grouper_freq)).apply("sum").reset_index()

    flight_nr_df = flight_nr_df.reindex(columns=flt_nums)
    flight_nr_df = flight_nr_df.fillna(0)
    flight_nr_df[flight_nr_df.columns[1:]] = flight_nr_df.iloc[:,1:].astype("int")

    flight_nr_df.reset_index(inplace=True, drop=True)

    return flight_nr_df

def get_weather_forecast():
    """
    returns next 96 values
    """
    x_time=time_now_15()

    start_dt = x_time
    end_dt = x_time + dt.timedelta(days = 1)


    weather_forecast_url = "https://opendata.fmi.fi/wfs?service=WFS&version=2.0.0&request=getFeature&storedquery_id=fmi::forecast::harmonie::surface::point::timevaluepair&place=helsinki&parameters=Temperature,Precipitation1h&timestep=15"
    response = requests.get(weather_forecast_url)
    xml_data = response.text

    root = ET.fromstring(xml_data)

    namespaces = {
    'wfs': "http://www.opengis.net/wfs/2.0",
    'omso': "http://inspire.ec.europa.eu/schemas/omso/3.0",
    'gml': "http://www.opengis.net/gml/3.2",
    'om': "http://www.opengis.net/om/2.0",
    'sams': "http://www.opengis.net/samplingSpatial/2.0",
    'sam': "http://www.opengis.net/sampling/2.0",
    'target': "http://xml.fmi.fi/namespace/om/atmosphericfeatures/1.1",
    'wml2': "http://www.opengis.net/waterml/2.0"
    }

    sää_df = pd.DataFrame(columns=["datetime", "value", "type"])

    for member in root.findall('wfs:member', namespaces):
        for observation in member.findall('omso:PointTimeSeriesObservation', namespaces):
            obs_type = observation.get('{http://www.opengis.net/gml/3.2}id').split("-")[-1]
            for point in observation.findall('.//wml2:MeasurementTVP', namespaces):
                time = point.find('wml2:time', namespaces).text
                value = point.find('wml2:value', namespaces).text
                sää_df.loc[len(sää_df)] = time, value, obs_type

    sää_df = sää_df.pivot(columns="type", values="value", index="datetime").reset_index().rename_axis(None, axis=1)
    
    sää_df["datetime"] = pd.to_datetime(sää_df["datetime"])
    sää_df["datetime"] = sää_df["datetime"].dt.tz_convert("Europe/Helsinki")
    sää_df["datetime"] = pd.to_datetime(sää_df["datetime"])
    sää_df["datetime"] = sää_df["datetime"].dt.tz_localize(None)
    sää_df = sää_df.rename(columns={"Precipitation1h":"rain", "Temperature":"temp"})
    sää_df["rain"] = sää_df["rain"].apply(lambda x: float(str(x).replace("-", "0")))
    sää_df["temp"] = sää_df["temp"].astype(float)

    sää_df.loc[len(sää_df.index)] = [sää_df.loc[0][0] - dt.timedelta(minutes=15), sää_df.loc[0][1], sää_df.loc[0][2]]

    sää_df = sää_df.sort_values("datetime", ascending=True)

    sää_df = sää_df[sää_df["datetime"]>=start_dt]
    sää_df = sää_df[sää_df["datetime"]<end_dt]

    sää_df = sää_df.reset_index(drop=True)

    return sää_df

###

def create_dfs_for_model(rides_df_, flights_df_, sää_df_, window_len=96, ME=False):
    window_len = window_len + 1
    forecast_df_features = rides_df_.copy()
    forecast_df_features = forecast_df_features.merge(sää_df_[:96], how="outer")
    forecast_df_features["time"] = forecast_df_features['datetime'].dt.hour * 60 + forecast_df_features['datetime'].dt.minute
    forecast_df_features["weekday"] = forecast_df_features["datetime"].dt.weekday

    features = [
        "datetime",
        "weekday",
        "time",
        "temp",
        "rain"
    ]

    features_flights_bool = True
    with open(f"{cwd}cols_not_flt.json","r") as f:
        cols_not_flt = json.load(f)
    cols_not_flt = cols_not_flt + ["datetime"]
    n_shifts = len(forecast_df_features) - window_len -1

    columns = [f'y_t-{i}' for i in range(window_len)]

    df_train = pd.DataFrame(columns=columns[::-1])
    df_train = df_train.rename(columns={"y_t-0":"y"})

    for i in range(n_shifts):
        df_train.loc[i] = forecast_df_features["sum"].values[i:window_len+i]

    for i in range(n_shifts):
        for feature in features:
            df_train.loc[i, feature] = forecast_df_features[feature][window_len+i]

    if "weekday" in features:
        df_train = pd.get_dummies(df_train, columns=["weekday"], dtype="int")
        df_train = df_train.reindex(columns=cols_not_flt)

    if features_flights_bool:
        df_train = df_train.merge(flights_df_, on="datetime", how="outer")
        
    df_train = df_train.fillna(0)

    df_preds = df_train[["datetime", "y"]]
    df_preds["y"] = np.nan

    df_train.drop(columns="datetime", inplace=True)
    df_train.reset_index(drop=True, inplace=True)

    df_train.iloc[1:,:97] = np.nan
    df_train = df_train.ffill()

    if ME:
        scaler = joblib.load(f"{cwd}scaler_ME.gz")
    else:
        scaler = joblib.load(f"{cwd}scaler.gz")

    df_train.iloc[:,:100] = scaler.transform(df_train.iloc[:,:100])

    return df_train, df_preds

def inverse_transform_preds(y, df_train, ME):
    y_pred_df = pd.DataFrame(columns=df_train.columns[:100])
    y_pred_df.iloc[:,0] = y

    if ME:
        scaler = joblib.load(f"{cwd}scaler_ME.gz")
    else:
        scaler = joblib.load(f"{cwd}scaler.gz")
    
    y_pred_df = pd.DataFrame(scaler.inverse_transform(y_pred_df))
    y_transformed = y_pred_df.iloc[:,0].values
    return y_transformed

# def import_models():
#     global models
#     models = []

#     print("Getting models...")
#     model_import_list = os.listdir("models/")
#     model_import_list = [int(x[7:-6]) for x in model_import_list]
#     model_import_list = sorted(model_import_list)
#     model_import_list = [f"model_t{x}.keras" for x in model_import_list]

#     print("Importing models...")
#     for file in model_import_list:
#         model = tf.keras.models.load_model(f"models/{file}")
#         models.append(model)
#         print(f"{file} imported!")

#     print("Models imported!")


def import_model(number, ME):
    filename = f"model_t{number}.keras"

    if ME:
        model = tf.keras.models.load_model(f"{cwd}models_ME/{filename}")   
    else:
        model = tf.keras.models.load_model(f"{cwd}models/{filename}")   

    return model

def predict_24h(df_train_, ME):
    y_preds = []

    for i in range(len(df_train_)):
        model_num = i%96 + 1
        # model = models[model_num]
        model = import_model(model_num, ME)
        X_test = df_train_.iloc[i:i+1].drop(["y"], axis=1)
        X_test = np.array(X_test)

        y_pred = model.predict(X_test, verbose=0, batch_size=256)
        y_pred = y_pred.reshape(1, -1)[0]
        y_preds.append(y_pred)
        del model

    preds = np.concatenate(y_preds)
    preds = inverse_transform_preds(preds, df_train=df_train_, ME=ME)
    preds = np.clip(preds, 0, 100)

    return preds

dtype_mapping_preds = {
    'datetime': DateTime,
    'y': Float
}

def job_preds(ME):
    try:
        print("Starting job_preds()...")
        t=time_now_15()

        rides_df = get_ride_data(t, ME=ME)
        flights_df = get_flight_data(t)
        sää_df = get_weather_forecast()

        df_train, _ = create_dfs_for_model(rides_df, flights_df, sää_df, ME=ME)
        preds = predict_24h(df_train, ME)
        preds = savgol_filter(preds, 5, 2)

        dt_range = pd.date_range(start=t, periods=len(preds), freq="15min")
        df_preds = pd.DataFrame({"datetime":dt_range, "y":preds})
        
        # keskiarvoista vanhojen kanssa
        if ME:
            df_preds_old = get_sql_table("preds_me")
        else:
            df_preds_old = get_sql_table("preds")

        df_preds_old = df_preds_old.rename(columns={"y":"y_old"})
        df_preds = df_preds.merge(df_preds_old, how="left")
        df_preds = df_preds.ffill()
        df_preds["y"] = df_preds.apply(lambda row: np.mean(row[1:]), axis=1)
        df_preds = df_preds.drop(columns=["y_old"])
        # # #

        print("New preds generated!")
    except Exception as error:
        if ME:
            df_preds = get_sql_table("preds_me")
        else:
            df_preds = get_sql_table("preds")

        df_preds = df_preds[1:].reset_index(drop=True)
        last_row = df_preds.iloc[-1:]
        df_preds = pd.concat([df_preds, last_row])
        print("AN ERROR OCCURED:", error)
        print("Copied old predictions to DB")

    if ME:
        df_preds.to_sql("preds_me", sql_engine, if_exists='replace', index=False, dtype=dtype_mapping_preds)
    else:
        df_preds.to_sql("preds", sql_engine, if_exists='replace', index=False, dtype=dtype_mapping_preds)

    del rides_df, flights_df, sää_df, df_train, df_preds
    gc.collect()    

job_preds(ME=True)