# Taxi demand prediction app
![banner_image](banner.jpg)

At Helsinki Airport, a taxi driver is almost guaranteed to find a customer. However, the number of arriving air passengers varies greatly depending on the day and time. The goal of this project was to determine the most optimal times for a taxi driver to queue at the airport taxi station.

Using data from Finavia (airport operator) and Taxipoint (airport taxi traffic coordinator), it is possible to predict the demand for the next 24 hours.

Real time data collection from APIs and a simple neural network prediction script were ran on AWS cloud. A new prediction was provided every 15 minutes. An app visualizing the predictions was hosted on Streamlit. 

- Website: https://taxipoint.streamlit.app/
- Github: https://github.com/pyrypp/taxipoint_streamlit

Data was collected for 6 months, and the prediction service was running for approximately one month. It was used by over 40 taxi drivers but was shut down due to high costs.

## Demo
This is a real timelapse of the service in operation. On the left side there are past observations. On the right side is the prediction. The unit of the y-axis is the number of customers per 15 minues.

Traffic peaks are highlighted and the number of passengers in that peak time is indicated on top of the peak.
[![demo_gif](/images/demo.gif)](https://taxipoint.streamlit.app/)

## How it works
At Helsinki Airport the main taxi station fits approximately 30 cars. To avoid queues on the public road leading to the airport, there is a designated queing area for taxis slightly further away. 

There are automatic boom barriers controlling the flow of taxi traffic. There is also a website operated by Taxipoint, which shows the number of cars at the taxi station and the queuing area.

When a car leaves the taxi station, a new car is free to enter.

![map_image](/images/airport_map_3_lq.jpg)
_Image: Google Maps_

**A data scraper** script was built with Python to track the website. When a car left the taxi station, it was logged as one customer. The script also monitored the queue length at the queue area.

Data on the arriving flights was also collected every day through Finavia's own API.

Data on taxi rides and arriving flights was collected for six months. All data was stored in a PostgreSQL **database** on AWS RDS. Weather data from Ilmatieteen laitos was also incorporated in the training of the model and the predictions.

The next step was to build a prediction **model** to predict the number of taxi rides for the next 24 hours. After trying several methods, such as a seasonal naive model, SARIMA and a random forest model, a simple neural network model was chosen as it proved the most accurate.

The model had two hidden layers (512 + 256 units) and outputted a single value. The 24-hour prediction was made with a resolution of 15 minutes. A total of 97 models were trained each predicting one step further into the future.

The model took as input ride data from the previous 24 hours and the scheduled flight arrivals and forecasted weather of the following 24 hours. A Python script was running on AWS EC2 calculating a new **prediction** every 15 minutes and storing the prediction in the database.

Another script rendered an image of a **plot** visualizing the prediction. This image was stored on AWS S3.

A **website** built was with Streamlit and hosted on Streamlit community cloud. It fetched the plot images from S3 and displayed them for the users.

![diagram_image](/images/diagram.png)
