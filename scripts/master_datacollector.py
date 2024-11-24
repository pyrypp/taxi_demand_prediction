
import pandas as pd
import requests
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
import time
import numpy as np
import json
import os
import re
import psycopg2
import xml.etree.ElementTree as ET
import schedule
import pytz
from sqlalchemy.types import *
from sqlalchemy import create_engine
import gc

# from predict import job_preds



print("Starting program...")

os.environ['TZ'] = 'Europe/Helsinki'
time.tzset()


areastats_url = "https://ajooikeudet.taxipoint.fi/VisyWebSale/areastats.do"

# PostgreSQL connection parameters
db_user = 'postgres'
db_password = '******' # hidden
db_host = '******' # hidden
db_port = '5432'
db_name = ''

# Create SQLAlchemy engine
sql_engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')



### TAXIPOINT - RIDES/QUEUE

def get_terminalstats():
    response = requests.get(areastats_url)
    response_string = response.text

    data_list = response_string.split(',')
    result_dict = dict(element.split('=') for element in data_list if '=' in element)

    FT = result_dict["t21"]
    TH = result_dict["t22"]
    ME = result_dict["t23"]
    MU = result_dict["t24"]

    return  FT, TH, ME, MU


def get_queuestats():
    response = requests.get(areastats_url)
    response_string = response.text

    data_list = response_string.split(',')[:4]
    data_list = [re.sub(r'/\w\w', "", element) for element in data_list]
    result_dict = dict(element.split('=') for element in data_list)

    FT_q = int(result_dict["k1"])
    TH_q = int(result_dict["k2"])
    ME_q = int(result_dict["k3"])
    MU_q = int(result_dict["k4"])

    return  FT_q, TH_q, ME_q, MU_q


def get_date_time():
    current_time = datetime.now().time()
    formatted_time = current_time.strftime("%H:%M:%S")

    current_date = datetime.now().date()
    formatted_date = current_date.strftime("%Y-%m-%d")

    return formatted_date, formatted_time



def save_to_sql(new_row, table_name):
    # Establish a connection to the PostgreSQL database
    try:
        with psycopg2.connect(
            dbname="",
            user="postgres",
            password="teslataksi",
            host="taxipoint-db.c948kg8sepos.eu-north-1.rds.amazonaws.com",
            port="5432"
        ) as conn:

            with conn.cursor() as cur:

                insert_query = 'INSERT INTO {} (date, time, "FT", "TH", "ME", "MU") VALUES (%s, %s, %s, %s, %s, %s)'.format(table_name)
                cur.execute(insert_query, (new_row[0], new_row[1], new_row[2], new_row[3], new_row[4], new_row[5]))

                conn.commit()
    except:
        None

def save_queuestats():
    questats_data = get_queuestats()
    formatted_date, formatted_time = get_date_time()
    new_row = [formatted_date, formatted_time] + list(questats_data)
    save_to_sql(new_row, "queue")



FT, TH, ME, MU = get_terminalstats()
t_mns_10_data = [FT, TH, ME, MU]

def job_taxipoint():
    global t_mns_10_data
    print("Data requested...")
    try:
        formatted_date, formatted_time = get_date_time()
        FT, TH, ME, MU = get_terminalstats()

        last_log_data = np.array(t_mns_10_data, dtype=np.int64)
        new_log_data = np.array([FT, TH, ME, MU], dtype=np.int64)

        new_row_data = list(new_log_data - last_log_data)
        new_row_data = [int(max(0,element)) for element in new_row_data]

        new_row = [formatted_date, formatted_time] + new_row_data

        t_mns_10_data = new_log_data

        if new_row_data != [0,0,0,0]:
            save_to_sql(new_row, "rides")
            print(f" -> Ride data logged - {formatted_date} - {formatted_time}")
        else:
            print(" -> No new ride data logged.")

    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        save_queuestats()
        print(" -> Queuestats logged")
    except Exception as e:
        print(f"!! Logging questats failed !! - {e}")

### PAX
        
def str_date_add_one_day(date):
    new_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
    return str(new_date.date())

pax_url = "https://ajooikeudet.taxipoint.fi/VisyWebSale/"

def get_pax_nums():
    response = requests.get(pax_url)
    response_string = response.text
    document = BeautifulSoup(response_string, "html.parser")

    pax_str = str(document.find(attrs={"class":"alert alert-secondary"}).find_all("p")[1])
    data_list = pax_str.split('<br/>')

    pattern = re.compile(r'.*?((\d+\s*-\s*\d+).*?(\d+)).*?')

    data_list = [item for item in data_list if pattern.match(item)]

    pattern1 = re.compile(r'\d\d:\d\d\s*-\s*\d\d:\d\d')
    pattern2 = re.compile(r'\d+$')
    pattern3 = re.compile(r'<[^>]*>')
    pattern4 = re.compile(r'[\r\n\t]+')
    data_list = [re.sub(pattern3, '', item) for item in data_list]
    data_list = [re.sub(pattern4, '', item) for item in data_list]

    data_list = [pattern1.findall("".join(item))[0] + " " + pattern2.findall("".join(item))[0] for item in data_list]
    data_list = [item.replace(" - ", "-") for item in data_list]
    data_list = [item.replace(":00", "") for item in data_list]

    data_list = [element.split() for element in data_list]

    pax_df = pd.DataFrame(data_list, columns=["time", "pax"])
    pax_df["time"] = pax_df["time"].apply(lambda x: x.split("-")[0])
    pax_df = pax_df.astype(int)

    formatted_date, formatted_time = get_date_time()
    pax_df["date"] = str(formatted_date)
    pax_df["date"] = np.where(pax_df["time"]==0, str_date_add_one_day(pax_df["date"][0]), pax_df["date"])

    pax_df = pax_df[["date", "time", "pax"]]
    
    return pax_df

dtype_mapping_pax = {
    'date': Date,
    'time': Integer,
    'pax': Integer
}

def job_pax():
    try:    
        date,_ = get_date_time()
        pax_df_return = get_pax_nums()
        # write_json(pax_df_return, date)
        pax_df_return.to_sql("pax", sql_engine, if_exists='append', index=False, dtype=dtype_mapping_pax)
        print(f"~~~~~~~~~~~~~~~~~~~~~ Pax data for {date} logged! ~~~~~~~~~~~~~~~~~~~~~")
    except:
        print("!! AN ERROR OCCURED !!")

# schedule.every().day.at("18:00").do(job_pax)



### FINAVIA

url_finavia      = 'https://api.finavia.fi/flights/public/v0/flights/arr/HEL'
headers_finavia  = {"Accept":"application/xml","app_id": "******", "app_key":"******"} # hidden


aircrafts_df = pd.read_csv("aircrafts.txt", delimiter=";", names=["Aircraft name", "ICAO", "IATA", "Capacity", "Country"])

aircraft_capacity_dict = dict(zip(aircrafts_df["IATA"], aircrafts_df["Capacity"]))

capacity_additions = {
    "359":350,
    "32B":102,
    "223":130,
    "32N":180,
    "7M8":172,
    "32Q":220,
    "789":246,
    "788":240,
    "CS3":137,
    "32A":180,
    "73J":189
}

aircraft_capacity_dict.update(capacity_additions)


def get_flight_data():
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

    flights_df["date"] = flights_df["datetime"].apply(lambda x: x.date).astype(str)
    flights_df["time"] = flights_df["datetime"].apply(lambda x: x.time).astype(str)
    flights_df = flights_df[["date", "time", "fltnr", "actype"]]

    flights_df["Capacity"] = flights_df["actype"].map(aircraft_capacity_dict)
    flights_df["Capacity"] = flights_df["Capacity"].fillna(0)
    flights_df["Capacity"] = flights_df["Capacity"].astype(int)

    return flights_df

dtype_mapping_finavia = {
    'date': Date,
    'time': Time,
    'fltnr': String,
    'actype': String,
    'Capacity': Integer
}

def job_finavia():
    date = str(datetime.now().date())
    df = get_flight_data()
    df = df[df["date"]==date].reset_index(drop=True)
    try:
        df.to_sql("flights", sql_engine, if_exists='append', index=False, dtype=dtype_mapping_finavia)
        print(f"~~~~~~~~~~~~~~~~~~~~~ Flight data for {date} logged! ~~~~~~~~~~~~~~~~~~~~~")
    except:
        print("ERROR - Flight data logging failed: SQL error")
    

schedule.every().day.at("10:00").do(job_finavia)


###
while True:
    job_taxipoint()
    schedule.run_pending()
    time.sleep(10)
    date_now, time_now = get_date_time()
    print(f"The program is running - {date_now} {time_now}")
    gc.collect()