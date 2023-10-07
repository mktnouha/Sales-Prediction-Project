# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 21:32:09 2023

@author: pc
"""


import datetime
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.colors as color
import base64
import zipfile
import os
import xgboost as xgb

# Define paths
zip_path = 'train.zip'
extracted_dir = 'extracted_file/'

# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

# Read the CSV file from the extracted directory
csv_path = os.path.join(extracted_dir, 'train.csv')

# Now you can work with the 'train' DataFrame

def load_data_and_model() :
    #model = pickle.load(open('xgb_model.pkl', 'rb'))
    model = xgb.Booster(model_file='xgb_model_exported.model')
    df = pd.read_csv('test.csv')
    oil = pd.read_csv('oil.csv')
    holidays = pd.read_csv('holidays_events.csv')
    stores = pd.read_csv('stores.csv')
    train = pd.read_csv(csv_path)
    return model, df, oil, holidays, stores, train

model, df, oil, holidays, stores, train = load_data_and_model()

unique_families = df['input_family'].unique()

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

df.drop(['id'], axis=1, inplace=True)

def create_features(line):
    line = line.copy()
    line['quarter'] = line.index.quarter
    line['month'] = line.index.month
    line['year'] = line.index.year
    line['weekofyear'] = line.index.isocalendar().week
    line['dayofyear'] = line.index.dayofyear
    return line    


def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
         f"""
        <style>
        .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
         }}
        </style>
        """,
        unsafe_allow_html=True
         )
add_bg_from_local('backgroundImage.jpg')
with st.sidebar: 

    selected = option_menu('Ecuador sales predictions',
                        ['Sales Prediction', 'Predicting Sold Products for a given store', 'Exploratory Data Analysis'],
                        icons=['star-half','list-stars','heart-half'],
                        default_index=0)



if (selected == 'Sales Prediction'):
    
    st.title('Sales prediction')
    
    selected_date = st.date_input("Enter the date:", datetime.datetime(2017, 8, 16))
    store_nbr_value = st.number_input("Enter the store number:", min_value=1, step=1, value=1, format="%d")
    selected_family = st.selectbox('Select a product family', unique_families)
    datetime_object = datetime.datetime.strptime(str(selected_date), '%Y-%m-%d')
    filtered_row = df.loc[(df.index == datetime_object) & (df['store_nbr'] == int(store_nbr_value)) & (df['input_family'] == str(selected_family))]
    filtered_row = create_features(filtered_row)
    

    if st.button('Predict '):

        
        FEATURES = ['family', 'store_nbr', 'onpromotion', 'dcoilwtico', 'cluster', 'national_holiday_type','state_holiday_type','city_holiday_type', 'quarter' ,'month', 'year', 'dayofyear']

        X_test = filtered_row[FEATURES]
        X_test_dmatrix = xgb.DMatrix(X_test)        
        prediction = model.predict(X_test_dmatrix)
        
        prediction = np.where(prediction < 0, 0, prediction)
        st.write("Predicted Sales:")

        st.success(str(prediction[0]))
        
if (selected == 'Predicting Sold Products for a given store'):
    st.title('Prediction of Sold Products for a given store')

    selected_date = st.date_input("Enter the date:", datetime.datetime(2017, 8, 16))
    datetime_object = datetime.datetime.strptime(str(selected_date), '%Y-%m-%d')
    
    store_nbr_value = st.number_input("Enter the store number:", min_value=1, step=1, value=1, format="%d")

    filtered_row = df.loc[(df.index == datetime_object) & (df['store_nbr'] == int(store_nbr_value))]
    filtered_row = create_features(filtered_row)

    FEATURES = ['family', 'store_nbr', 'onpromotion', 'dcoilwtico', 'cluster', 'national_holiday_type', 'state_holiday_type', 'city_holiday_type', 'quarter', 'month', 'year', 'dayofyear']

    X_test = filtered_row[FEATURES]
    X_test_dmatrix = xgb.DMatrix(X_test) 

    if st.button('Sold Products'):
        prediction = model.predict(X_test_dmatrix)
        prediction = np.where(prediction < 0, 0, prediction)

        family_prediction = pd.DataFrame({'Product': filtered_row['input_family'], 'predicted sales': prediction.flatten()})
        family_prediction = family_prediction[family_prediction['predicted sales'] > 0]
        family_prediction = family_prediction.reset_index(drop=True)
        st.write(family_prediction)

if (selected == 'Exploratory Data Analysis'):

    st.title('Exploratory data analysis')

    oil['date'] = pd.to_datetime(oil['date'])
    train['date'] = pd.to_datetime(train['date'])

    dates = pd.DataFrame(train.date.unique(), columns=['date'])
    dates['date'] = pd.to_datetime(dates['date'])
    oil = pd.merge(dates, oil, how="left", on=['date'])
    oil = oil.set_index('date')
    # Create the heatmap figure using seaborn
    fig7 = sns.heatmap(oil.isna(), cbar=False)
    fig7.set_title("Heat map for oil prices (Checking missing values distribution)")

        # Display the figure using Streamlit
    st.pyplot(fig7.figure)
    

    oil['dcoilwtico'][0] = oil['dcoilwtico'][1].copy()
    oil = oil.dcoilwtico.interpolate(method='linear')

    # Create the plot using Plotly
    fig1 = px.line(oil, x=oil.index, y='dcoilwtico', title='Plot slider for oil prices')
    fig1.update_xaxes(rangeslider_visible=True)
    fig1.update_layout(height=600, width=900)

    # Display the plot on Streamlit
    st.plotly_chart(fig1)

    st.markdown("##")

    train = pd.merge(train, oil, how="left", on=['date'])
    grouped = train.groupby(['date']).agg({'dcoilwtico': 'mean', 'sales': 'mean'})
    fig2 = px.line(grouped, x=grouped.index , y=['sales', 'dcoilwtico'])
    fig2.update_xaxes(rangeslider_visible=True)
    fig2.update_layout(title='Sale and Date Plot')
    fig2.update_layout(height=600, width=900)
    st.plotly_chart(fig2)


    # Generate the Seaborn bar plot
    grouped_data = train.groupby('family').sales.sum().reset_index()

   # Sort the DataFrame by 'sales' in descending order
    grouped_data = grouped_data.sort_values('sales', ascending=False)

# Create the bar plot using Plotly Express
    fig3 = px.bar(grouped_data, x='family', y='sales', orientation='v')
    fig3.update_layout(title='Total Sales by Family of Products')
    fig3.update_layout(height=600, width=900)


    # Display the family sales bar plot on Streamlit
    st.plotly_chart(fig3)
    
    
    # Display the transferred holidays analysis plot
    holiday_type = dict(holidays["type"].value_counts())
    transferred = holidays["transferred"].value_counts()
    transferred_dict = {"True": transferred[True], "False": transferred[False]}

    #plt.figure(figsize=(20, 8))
    left = plt.subplot(1, 2, 1)
    plt.bar(x=transferred_dict.keys(), height=transferred_dict.values(), color=["#008ae6", "#99d6ff"])
    for key in transferred_dict.keys():
       plt.annotate(f"{transferred_dict[key]}", (key, transferred_dict[key] + 3), size=10)
    plt.xticks(size=10)
    left.set_title("Transferred holidays distribution\n", size=13)
    left.set_xlabel("Transferred or not", size=12)
    left.set_ylabel("Total", size=12)
    

   
    right2 = plt.subplot(1, 2, 2)
    holiday_locale = dict(holidays["locale"].value_counts())
    colors = sns.color_palette("Set2", len(holiday_locale))
    plt.bar(x=list(holiday_locale.keys()), height=list(holiday_locale.values()), color=colors)
    for key in holiday_locale.keys():
       plt.annotate(f"{holiday_locale[key]}", (key, holiday_locale[key] + 3), size=10)
    plt.xticks(size=10)

    right2.set_title('Holiday type Counts\n', size=13)
    right2.set_xlabel('Type', size=12)
    right2.set_ylabel('Count', size=12)
    # Adjust spacing between plots
    fig4=plt.subplots_adjust(wspace=0.5)

# Display the transferred holidays analysis plot on Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)


    st.pyplot(fig4)
   # pie de promotion en fct des categories

# Calculate the mean sales and on promotion for each family category
    data_grouped_family_types = train.groupby('family')[['sales', 'onpromotion']].mean().reset_index()

# Sort the data by mean on promotion in descending order
    data_grouped_family_types.sort_values('onpromotion', ascending=False, inplace=True)

# Get the top 33 categories
    top_33_family_types = data_grouped_family_types.head(33)

# Calculate the percentage
    total_onpromotion = top_33_family_types['onpromotion'].sum()
    top_33_family_types['%_p'] = 100 * top_33_family_types['onpromotion'] / total_onpromotion
    top_33_family_types['%_p'] = top_33_family_types['%_p'].round(decimals=3)

# Get the labels and percentages
    labels = top_33_family_types['family']
    percentages = top_33_family_types['%_p']

# Get a list of 33 distinct colors from Plotly's colorscale
    color_palette = color.qualitative.Plotly[:33]

# Create the pie chart
    fig5, ax = plt.subplots()
    ax.pie(percentages, startangle=90, radius=1.5, colors=color_palette)

# Set labels for the pie slices
    labels_with_percentages = ['{0} - {1:1.2f} %'.format(label, percentage) for label, percentage in zip(labels, percentages)]
    ax.legend(labels_with_percentages, loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=8)
    plt.title(" Promotion by Family Category\n\n\n")# Display the pie chart in Streamlit

    st.pyplot(fig5)
    # Compute the correlation matrix

    # Load your data or use the 'train' DataFrame as an example
    #df = pd.read_csv(r'C:\Users\pc\Downloads\forpairplot.csv')
    #train_numeric = df.apply(pd.to_numeric, errors='coerce')
  

# Group the DataFrame by 'store_nbr' and calculate the sum of 'onpromotion' and 'sales'
    grouped_data = train.groupby('store_nbr')[['onpromotion', 'sales']].sum().reset_index()

# Create the scatter plot using Plotly Express
    fig6 = px.scatter(grouped_data, x='onpromotion', y='sales', title='Promotion and Sales Relationship')

# Display the plot on Streamlit
    st.plotly_chart(fig6)
















    
