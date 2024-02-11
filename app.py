import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from keras.models import load_model
import streamlit as st
from PIL import Image
import database as db
import pickle 
from pathlib import Path
import streamlit_authenticator as stauth  # pip install streamlit-authenticator


import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from tensorflow import keras
import pandas_datareader as data
from keras import layers
import yfinance as fn
import datetime
import cufflinks as cf

st.set_page_config(
    page_icon="👋",
)
  
  
  
  # --- USER AUTHENTICATION ---
names = ["Peter Parker", "Rebecca Miller"]
usernames = ["pparker", "rmiller"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "stock_price", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    # ---- READ EXCEL ----
    st.title('Stock Price Prediction')
    st.sidebar.header("Navigation bar")
    df=pd.read_csv(r"C:\Users\user\systempr\bitcoin.csv", index_col='Date', parse_dates=True, usecols=['Date','Close'])

    uploaded_file = st.file_uploader("Upload Files",type=['csv'])

    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        st.write(file_details)
        df=pd.read_csv(uploaded_file,index_col='Date', parse_dates=True, usecols=['Date','Close'])
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")
    user_input=st.text_input('Enter Stock Ticker','Enter Ticker')


    def get_labelled_windows(x, horizon=1):
            return x[:, :-horizon], x[:, -horizon:]

    def make_windows(x, window_size=7, horizon=1):
        window_step = np.expand_dims(np.arange(window_size + horizon), axis = 0)
        window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis = 0).T

        windowed_array = x[window_indexes]
        windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
        print(type(windows))
        return windows, labels
    def make_preds(model, input_data):
        forecast = model.predict(input_data)
        return tf.squeeze(forecast)
    if(user_input =="" or user_input =='Enter Ticker'):
        st.write('Please choose the stock you want to predict')
    elif(user_input !='bitcoin' and user_input is not None):
        st.sidebar.subheader('Query Parameters')
        start=st.sidebar.date_input("Start date",datetime.date(2010,1,1))
        end=st.sidebar.date_input("End date",datetime.date(2019,12,31))
        df2=fn.download(user_input,start,end)
        df2.to_csv('df21.csv',index=True)
        #st.write(df2)
        df=pd.read_csv(r'df21.csv', parse_dates=True, usecols=['Close'])
        ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
        tickerSymbol =  user_input
        c1=fn.Ticker(tickerSymbol)
        
        string_logo = '<img src=%s>' % c1.info['logo_url']
        st.markdown(string_logo, unsafe_allow_html=True)
        string_name = c1.info['longName']
        st.header('**%s**' % string_name)
        
        st.write(c1.info['longBusinessSummary'])
        st.subheader('Data within the given range')
        st.write(df2)
        st.line_chart(df2.describe())
        
        # Bollinger bands
        st.header('**Bollinger Bands**')
        qf=cf.QuantFig(df2,title='First Quant Figure',legend='top',name='GS')
        qf.add_bollinger_bands()
        fig = qf.iplot(asFigure=True)
        st.plotly_chart(fig)
        
        user_input2=st.number_input('Enter split_size',0.2)
        split_size = int(user_input2 * len(df))
        testdf=df.iloc[split_size: , : ]
        df=df.iloc[ :split_size, : ]
        st.subheader('shape of train data')
        st.write(df.shape)
        st.subheader('figure of train data')
        fig=plt.figure(figsize=(12,6))
        plt.plot(df)
        st.pyplot(fig)
        st.subheader('shape of test data')
        st.write(testdf.shape)
        st.subheader('figure of test data')
        fig=plt.figure(figsize=(12,6))
        plt.plot(testdf)
        st.pyplot(fig)


        adf = adfuller(df.Close)
        st.write(adf[1])

        d=0
        while(adf[1]>0.05):
            dummy=df.diff().dropna()
            d+=1
            adf = adfuller(dummy.Close)

        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
        plot_acf(df, lags=15, zero=False, ax=ax1)
        plot_pacf(df, lags=15, zero=False, ax=ax2)
        plt.show()
        st.pyplot(fig)


        fig,(ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
        plot_acf(df.diff().dropna(), lags=15, zero=False, ax=ax1)
        plot_pacf(df.diff().dropna(), lags=15, zero=False, ax=ax2)
        plt.show()
        st.pyplot(fig)



        order_aic_bic =[]
        for p in range(3):
        
        # Loop over q values from 0-3
            for q in range(3):
                try:
                    # Create and fit ARMA(p,q) model
                    model = ARIMA(df, order=(p,d,q))
                    results = model.fit()
                    
                    # Print p, q, AIC, BIC
                    order_aic_bic.append((p, q, results.aic))
                    
                except:
                    order_aic_bic.append((p, q, None))

        order_df = pd.DataFrame(order_aic_bic, columns=['p','q','aic'])

        p = order_df.sort_values('aic')[0:1].p.values[0]
        q = order_df.sort_values('aic')[0:1].q.values[0]
        if p==0 and q==0:
            p = order_df.sort_values('aic')[1:2].p.values[0]
            q = order_df.sort_values('aic')[1:2].q.values[0]


        #fig=plt.figure(figsize=(14,8))
        model = ARIMA(df,order=(p,d,q))
        results = model.fit()
        results.plot_diagnostics()
        #plt.show()
        #st.pyplot(fig,'b')
        fig=plt.figure(figsize=(10,6))
        forecast = results.get_forecast(steps=293)
        mean_forecast = forecast.predicted_mean
        plt.plot(df.index, df.Close, label='observed')
        plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')
        plt.xlabel('Date')
        plt.ylabel('Bitcoin Price - USD')
        plt.legend()
        plt.show()
        st.pyplot(fig)

        fig=plt.figure(figsize=(10,6))
        mae = np.mean(np.abs(results.resid))
        newdf = df.iloc[1:,:].Close + results.resid[1:].values
        newdf.plot()
        plt.show()
        st.pyplot(fig)

        fig=plt.figure(figsize=(10,6))
        ax=newdf.plot()
        df.plot(ax=ax)
        plt.show()
        st.pyplot(fig)

        d=0
        while(adf[1]>0.05):
            dummy=testdf.diff().dropna()
            d+=1
            adf = adfuller(dummy.Close)

        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
        plot_acf(testdf, lags=15, zero=False, ax=ax1)
        plot_pacf(testdf, lags=15, zero=False, ax=ax2)
        plt.show()
        st.pyplot(fig)
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
        plot_acf(testdf.diff().dropna(), lags=15, zero=False, ax=ax1)
        plot_pacf(testdf.diff().dropna(), lags=15, zero=False, ax=ax2)
        plt.show()  
        st.pyplot(fig)

        order_aic_bic =[]
        for p in range(3):
        
        # Loop over q values from 0-3
            for q in range(3):
                try:
                    # Create and fit ARMA(p,q) model
                    model = ARIMA(testdf, order=(p,d,q))
                    results = model.fit()
                    # Print p, q, AIC, BIC
                    order_aic_bic.append((p, q, results.aic))
                except:
                    order_aic_bic.append((p, q, None))
        order_df = pd.DataFrame(order_aic_bic, columns=['p','q','aic'])

        p = order_df.sort_values('aic')[0:1].p.values[0]
        q = order_df.sort_values('aic')[0:1].q.values[0]
        if p==0 and q==0:
            p = order_df.sort_values('aic')[1:2].p.values[0]
            q = order_df.sort_values('aic')[1:2].q.values[0]


        model_test = ARIMA(testdf,order=(p,d,q))
        results_test = model_test.fit()
        results_test.plot_diagnostics()
        #plt.show()  
        fig=plt.figure(figsize=(10,6))
        newtestdf = testdf.iloc[1:,:].Close + results_test.resid[1:].values
        ax=newtestdf.plot()
        testdf.plot(ax=ax)
        plt.show()
        st.pyplot(fig)

        prices = newdf.values
        timesteps = tf.convert_to_tensor(newdf.index.values.astype(np.int64))
        timestepstest = tf.convert_to_tensor(testdf.index.values.astype(np.int64))
        X_train, y_train = newdf.index, prices
        X_test, y_test = newtestdf.index, newtestdf.values

        HORIZON = 1 
        WINDOW_SIZE = 7
        
        train_windows, train_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
        test_windows, test_labels = make_windows(y_test, window_size=WINDOW_SIZE, horizon=HORIZON)

        ##load the model
        model_7=load_model('keras_model.h5')
        

        model_7_preds = make_preds(model_7, test_windows).numpy()
        model_7_preds=np.append(model_7_preds,test_labels[-1])
        test_labels = np.insert(test_labels,0,model_7_preds[0])


        fig=plt.figure(figsize=(15,7))
        plt.plot(X_test[-len(test_labels):], test_labels,label='Test')
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.plot(X_test[-len(model_7_preds):], model_7_preds,label='Predictions')
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend(fontsize=10)
        st.pyplot(fig)


        fig=plt.figure(figsize=(15,7))
        plt.plot(X_test[-len(test_labels):-1], testdf[7:-1].Close.to_numpy(),label='Test')
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.plot(X_test[-len(model_7_preds):-1], model_7_preds[:-1],label='Predictions')
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend(fontsize=10)
        st.pyplot(fig)
        st.subheader('Root mean square error showing')
        rmse = np.sqrt(np.mean(model_7_preds[:-1] - testdf[7:-1].Close.to_numpy())**2)
        st.write(rmse)


        st.subheader('range given for next days prediction')
        user_input2=st.number_input('Enter range',1)
        st.write(model_7.predict(test_windows[0:user_input2]).ravel())


        fig=plt.figure(figsize=(15,7))
        plt.plot(X_test[-(user_input2+1):-1], testdf[7:user_input2+7].Close.to_numpy(),label='Test')
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.plot(X_test[-(user_input2+1):-1], model_7_preds[0:user_input2],label='Predictions')
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend(fontsize=10)
        st.pyplot(fig)
        
        st.subheader('Loss vs epoch curve')
        image1 = Image.open('C:/Users/user/systempr/pages/ima.jpg')
        #displaying the image on streamlit app
        st.image(image1, caption=None, width=None, use_column_width=None, clamp=False,channels='RGB', output_format='auto')
            

        
        




    elif(user_input =='bitcoin'):
        
        df=df.iloc[::-1]
        st.subheader('Data from 2018-2022')
        isha=st.slider('please choose a value',0,50,4)
        ok=st.button('displaying top values ')
        if ok:
            st.write('First given values')
            st.write(df.head(isha))
        isha1=st.slider('please choose a value',0,50,3) 
        ok1=st.button('displaying last values')
        if ok1:
            st.write('Last given values')    
            st.write(df.tail(isha1))
        st.write(df.describe())
        st.subheader('Closing Price vs time chart')
        fig=plt.figure(figsize=(12,6))
        plt.plot(df.Close)
        st.pyplot(fig)
        
        user_input2=st.number_input('Enter split_size',0.2)
        split_size = int(user_input2 * len(df))
        testdf=df.iloc[split_size: , : ]
        df=df.iloc[ :split_size, : ]
        st.subheader('shape of train data')
        st.write(df.shape)
        st.subheader('figure of train data')
        fig=plt.figure(figsize=(12,6))
        plt.plot(df)
        st.pyplot(fig)
        st.subheader('shape of test data')
        st.write(testdf.shape)
        st.subheader('figure of test data')
        fig=plt.figure(figsize=(12,6))
        plt.plot(testdf)
        st.pyplot(fig)


        adf = adfuller(df.Close)
        st.write(adf[1])

        d=0
        while(adf[1]>0.05):
            dummy=df.diff().dropna()
            d+=1
            adf = adfuller(dummy.Close)

        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
        plot_acf(df, lags=15, zero=False, ax=ax1)
        plot_pacf(df, lags=15, zero=False, ax=ax2)
        plt.show()
        st.pyplot(fig)


        fig,(ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
        plot_acf(df.diff().dropna(), lags=15, zero=False, ax=ax1)
        plot_pacf(df.diff().dropna(), lags=15, zero=False, ax=ax2)
        plt.show()
        st.pyplot(fig)



        order_aic_bic =[]
        for p in range(3):
        
        # Loop over q values from 0-3
            for q in range(3):
                try:
                    # Create and fit ARMA(p,q) model
                    model = ARIMA(df, order=(p,d,q))
                    results = model.fit()
                    
                    # Print p, q, AIC, BIC
                    order_aic_bic.append((p, q, results.aic))
                    
                except:
                    order_aic_bic.append((p, q, None))

        order_df = pd.DataFrame(order_aic_bic, columns=['p','q','aic'])

        p = order_df.sort_values('aic')[0:1].p.values[0]
        q = order_df.sort_values('aic')[0:1].q.values[0]
        if p==0 and q==0:
            p = order_df.sort_values('aic')[1:2].p.values[0]
            q = order_df.sort_values('aic')[1:2].q.values[0]


        #fig=plt.figure(figsize=(14,8))
        model = ARIMA(df,order=(p,d,q))
        results = model.fit()
        results.plot_diagnostics()
        #plt.show()
        #st.pyplot(fig,'b')
        fig=plt.figure(figsize=(10,6))
        forecast = results.get_forecast(steps=293)
        mean_forecast = forecast.predicted_mean
        plt.plot(df.index, df.Close, label='observed')
        plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')
        plt.xlabel('Date')
        plt.ylabel('Bitcoin Price - USD')
        plt.legend()
        plt.show()
        st.pyplot(fig)

        fig=plt.figure(figsize=(10,6))
        mae = np.mean(np.abs(results.resid))
        newdf = df.iloc[1:,:].Close + results.resid[1:].values
        newdf.plot()
        plt.show()
        st.pyplot(fig)

        fig=plt.figure(figsize=(10,6))
        ax=newdf.plot()
        df.plot(ax=ax)
        plt.show()
        st.pyplot(fig)

        d=0
        while(adf[1]>0.05):
            dummy=testdf.diff().dropna()
            d+=1
            adf = adfuller(dummy.Close)

        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
        plot_acf(testdf, lags=15, zero=False, ax=ax1)
        plot_pacf(testdf, lags=15, zero=False, ax=ax2)
        plt.show()
        st.pyplot(fig)
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
        plot_acf(testdf.diff().dropna(), lags=15, zero=False, ax=ax1)
        plot_pacf(testdf.diff().dropna(), lags=15, zero=False, ax=ax2)
        plt.show()  
        st.pyplot(fig)

        order_aic_bic =[]
        for p in range(3):
        
        # Loop over q values from 0-3
            for q in range(3):
                try:
                    # Create and fit ARMA(p,q) model
                    model = ARIMA(testdf, order=(p,d,q))
                    results = model.fit()
                    # Print p, q, AIC, BIC
                    order_aic_bic.append((p, q, results.aic))
                except:
                    order_aic_bic.append((p, q, None))
        order_df = pd.DataFrame(order_aic_bic, columns=['p','q','aic'])

        p = order_df.sort_values('aic')[0:1].p.values[0]
        q = order_df.sort_values('aic')[0:1].q.values[0]
        if p==0 and q==0:
            p = order_df.sort_values('aic')[1:2].p.values[0]
            q = order_df.sort_values('aic')[1:2].q.values[0]


        model_test = ARIMA(testdf,order=(p,d,q))
        results_test = model_test.fit()
        results_test.plot_diagnostics()
        #plt.show()  
        fig=plt.figure(figsize=(10,6))
        newtestdf = testdf.iloc[1:,:].Close + results_test.resid[1:].values
        ax=newtestdf.plot()
        testdf.plot(ax=ax)
        plt.show()
        st.pyplot(fig)

        prices = newdf.values
        timesteps = tf.convert_to_tensor(newdf.index.values.astype(np.int64))
        timestepstest = tf.convert_to_tensor(testdf.index.values.astype(np.int64))
        X_train, y_train = newdf.index, prices
        X_test, y_test = newtestdf.index, newtestdf.values

        HORIZON = 1 
        WINDOW_SIZE = 7
        
        train_windows, train_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
        test_windows, test_labels = make_windows(y_test, window_size=WINDOW_SIZE, horizon=HORIZON)

        ##load the model
        model_7=load_model('keras_model.h5')
        

        model_7_preds = make_preds(model_7, test_windows).numpy()
        model_7_preds=np.append(model_7_preds,test_labels[-1])
        test_labels = np.insert(test_labels,0,model_7_preds[0])


        fig=plt.figure(figsize=(15,7))
        plt.plot(X_test[-len(test_labels):], test_labels,label='Test')
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.plot(X_test[-len(model_7_preds):], model_7_preds,label='Predictions')
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend(fontsize=10)
        st.pyplot(fig)


        fig=plt.figure(figsize=(15,7))
        plt.plot(X_test[-len(test_labels):-1], testdf[7:-1].Close.to_numpy(),label='Test')
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.plot(X_test[-len(model_7_preds):-1], model_7_preds[:-1],label='Predictions')
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend(fontsize=10)
        st.pyplot(fig)
        st.subheader('Root mean square error showing')
        rmse = np.sqrt(np.mean(model_7_preds[:-1] - testdf[7:-1].Close.to_numpy())**2)
        st.write(rmse)


        st.subheader('range given for next days prediction')
        user_input2=st.number_input('Enter range',1)
        st.write(model_7.predict(test_windows[0:user_input2]).ravel())


        fig=plt.figure(figsize=(15,7))
        plt.plot(X_test[-(user_input2+1):-1], testdf[7:user_input2+7].Close.to_numpy(),label='Test')
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.plot(X_test[-(user_input2+1):-1], model_7_preds[0:user_input2],label='Predictions')
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend(fontsize=10)
        st.pyplot(fig)
        
        st.subheader('Loss vs epoch curve')
        image1 = Image.open('C:/Users/user/systempr/pages/ima.jpg')
        #displaying the image on streamlit app
        st.image(image1, caption=None, width=None, use_column_width=None, clamp=False,channels='RGB', output_format='auto')
            

