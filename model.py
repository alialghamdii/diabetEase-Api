
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
from collections import defaultdict as dic
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
import string
import matplotlib.pyplot as plt
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


def openCSV(filename):
    global columns, df
    columns = ["Info", "Device", "Date&Time", "Record_Type", "Value"]
    df = pd.read_csv(filename, encoding='latin-1', header = None, names = columns)
    return df

def predict(filename):
    print("DataGenerator Called")

    df = openCSV(filename)
    lst_col = "Info"

        
    df.drop(columns=["Device", "Record_Type"], axis= 1, inplace = True)
    df.drop(df.index[[0,1]], inplace=True)


    from functools import partial

    df['Value']=df['Value'].astype(float)
    df['Date&Time'] = pd.to_datetime(df['Date&Time'], format="%d-%m-%Y %H:%M")

    f = partial(pd.to_datetime, dayfirst=True)
    df2 = df[(df['Date&Time'] > f("20-06-2021 00:00")) & (df['Date&Time'] < f("20-06-2021 23:59"))]


    # to accomedate missing data
    df2 = df2.interpolate()

    # Using the previous Value predict the next one
    # using simple model that gives predicted value using 
    # last observed value for next reading
    p_df = df2.copy()
    p_df['Prediction'] = df['Value'].shift(-1)

    # This only serves as a placeholder, until more accurate
    # prediction models are introduced 


    from datetime import timedelta, datetime

    # we make sure to automate the process for every dataset
    # we need to choose certain days data to seperate 
    # and train and then test the rest


    # seperate date and time so we can use them
    df['Date'] = [d.date() for d in df['Date&Time']]
    df['Time'] = [d.time() for d in df['Date&Time']]


    # So first we get the number of days for our dataset
    days_count = (df['Date&Time'].iloc[-1] -  df[['Date&Time']].iloc[0]).dt.days

    # how many days we have
    print("Days count: ", days_count)
    # how many total readings we have
    print("Total Samples count: ", len(df["Value"]))

    # get start and end date for data

    # initial date
    date = str(df['Date'].iloc[0])
    endDate = str(df['Date'].iloc[-1])
    print("Start date for data: ", date)
    print("End date for data: ", endDate)
    print("\n")

    # Format date to fetch each day's data
    datetime_begin = (date + " 00:00")
    datetime_end = (date + " 23:59")
    datetime_DayBegin = datetime.strptime(datetime_begin, '%Y-%m-%d %H:%M')
    datetime_DayEnd = datetime.strptime(datetime_end, '%Y-%m-%d %H:%M')

    date = datetime_DayBegin

    # information on data that we have
    for i in range(int(days_count)):
        
        # convert date to string for printing
        current_date = datetime_DayBegin.strftime('%Y-%m-%d %H:%M')
        print('Day: ', current_date[0:9])
        
        # now we get how many readings for this day
        readings_perday = df[(df['Date&Time'] > f(datetime_DayBegin)) & (df['Date&Time'] < f(datetime_DayEnd))]
        print('readings per this day: ', len(readings_perday))
        
        # Keep formatting date
        datetime_DayBegin += timedelta(days=1)
        datetime_DayEnd += timedelta(days=1)
        
        # move to the next day
        date += timedelta(days=1)




    # Now we can determine our training and test data 
    # by getting readings from day and train 3/4 of the data

    # First we will treat the data as consecutive data
    # by not taking days into account

    samples = len(p_df["Value"])
    train_samples = int((samples * 3/4))
    print("Size of Training Sample is: ", train_samples)

    train = p_df[:-train_samples]

    test = p_df[-train_samples:]
    test_samples = samples - train_samples
    print("Size of Testing Sample is:  ", test_samples - 1)


    test = test.drop(test.tail(1).index) # Drop last row as it is NaN

    test = test.copy()

    # predicted value for this row (naive)
    test['pred_baseline'] = test['Value']



    # For this block I will put in varius ML methods to 
    # add to and compare


    # these methods are from https://towardsdatascience.com/
    # the-complete-guide-to-time-series-forecasting-using-sklearn-pandas-
    # and-numpy-7694c90e45c1

    # get rid of NaN Values
    train = train.dropna()
    test = test.dropna()

    # method 1

    X_train = train['Value'].values.reshape(-1,1)
    y_train = train['Prediction'].values.reshape(-1,1)
    X_test = test['Value'].values.reshape(-1,1)
    # Initialize the model
    dt_reg = DecisionTreeRegressor(random_state=42)
    # Fit the model
    dt_reg.fit(X=X_train, y=y_train)
    # Make predictions
    dt_pred = dt_reg.predict(X_test)
    # Assign predictions to a new column in test
    test['dt_pred'] = dt_pred

    # method 2

    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y=y_train.ravel())
    gbr_pred = gbr.predict(X_test)
    test['gbr_pred'] = gbr_pred
    
    prediction_data = test.to_csv("/csv_files/prediction_data.csv", encoding='utf-8', index=False)
    return prediction_data