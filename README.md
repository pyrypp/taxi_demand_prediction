# Taxi demand prediction app
![image](banner.jpg)

At Helsinki Airport, a taxi driver is almost guaranteed to find a customer. However, the number of arriving air passengers varies greatly depending on the day and time. The goal of this project was to determine the most optimal times for a taxi driver to queue at the airport taxi station.

Using data from Finavia and Taxipoint, which operates taxi traffic at the airport, it is possible to predict the demand for the next 24 hours.

Real time data collection from APIs and a simple neural network prediction script were ran on AWS cloud. A new prediction was provided every 15 minutes. An app visualizing the predictions was hosted on Streamlit. (https://github.com/pyrypp/taxipoint_streamlit)

Data was collected for 6 months, and the prediction service was running for approximately one month. It was used by over 40 taxi drivers but was shut down due to high costs.

![image](/images/plot_sum.png)
