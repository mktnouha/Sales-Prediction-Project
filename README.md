# Sales-Prediction-Project
# Sales prediction using ML.

Within this project, our goal is to anticipate the sales of 33 product families available in Favorita stores situated in Ecuador. Through the application of sophisticated data analysis methods and machine learning, we are striving to construct a dependable and precise sales prediction model.

## Dataset

The dataset is sourced from Kaggle : [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data?select=test.csv)


## Exploratory Data Aanalysis

Preliminary analyses were conducted to:

- derive intriguing insights regarding the interplay between the features and sales data
- Identify significant features to keep and features to remove.


## Modeling 

Using an XGBRegressor. trained on approximately 80% of the data, encompassing purchases made between 01-01-2013 and 02-01-2017. The remaining data, around 20%, is set aside for testing, spanning the period from 02-01-2017 to 08-15-2017. 

After evaluating the model and being satisfied with the results, we proceeded to retrain the model using the entirety of the available data. This enables us to maximize the utilization of the available information and enhance the model's performance further.


## Deployment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sales-prediction-project.streamlit.app/)


To enhance interaction with the model, we've developed a user-friendly graphical interface using Streamlit.

This interface consists of three pages:

- The first page of the interface enables users to predict the sales of a specific product category in a given store on a specific date.
- The second page of the interface facilitates predicting the products sold for a specific store on a given date.
- The final page of the interface incorporates several graphs that facilitate exploratory analysis of the available data.

