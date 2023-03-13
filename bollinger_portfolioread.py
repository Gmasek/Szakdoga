import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import datetime as dt
import pandas_datareader.data as web
from sklearn.linear_model import LinearRegression
import csv
import openpyxl

tickers =pd.read_excel("Porfolio.xlsx")['Tickers']
ROI = 0
workbook = openpyxl.Workbook()
worksheet = workbook.active
col = 67
row = 1


for ticker in tickers[0:20]:
    worksheet[chr(65) + str(row)]= ticker
    def Calc_bands(df,period):
        
        #print(df1.tail())
        df['ma'] = df['Adj Close'].rolling( window = period, min_periods=0).mean()
        df['stdiv']= df["Adj Close"].rolling(window = period , min_periods = 0).std()
        df['UpperBand'] = df['ma'] + (2 * df['stdiv'])
        df['LowerBand'] = df['ma'] - (2 * df['stdiv'])
        #print(df[['Adj Close','20ma','LowerBand', 'UpperBand']].tail())
        df['Adj Close'].plot()
        df['UpperBand'].plot()
        df['LowerBand'].plot()
       
        df['elad'] = np.where((df['Adj Close'] > df['UpperBand']) , 100, 0)
        df['Vesz'] = np.where((df['Adj Close'] < df['LowerBand']) , -100, 0)
        #df['elad'].plot()
        #df['Vesz'].plot()
        #plt.show()
          
    def buysell(dataframe, invested_amount):
        
        Qty = invested_amount / dataframe['Adj Close'][0]
        return_amount = 0
        i=1
        for ind in dataframe.index:
            i+=1
            if (dataframe['elad'][ind] == 1) & (Qty != 0) :
                return_amount = Qty * dataframe['Adj Close'][ind]
                Qty = 0
            if (dataframe['Vesz'][ind] == 1) & (return_amount != 0 ):  
                Qty = return_amount /  dataframe['Adj Close'][ind]
                return_amount = 0  
            worksheet[chr(col) + str(i)]= (max(return_amount,Qty*dataframe['Adj Close'][ind]))
        return return_amount, Qty        
    
    def getReturn(code , amount ):
        start_date =dt.datetime(2021,1,1)
        end_date =dt.datetime(2022,1,1)
        dataframe = yf.download(code,start_date,end_date, interval='1d',prepost="false")
        Calc_bands(dataframe,20)
        return_amount, Qty   = buysell(dataframe,amount)
        CurrentValue = Qty * dataframe['Adj Close'][-1]
        return max(return_amount,Qty,CurrentValue)
    
    ROI+=(getReturn(ticker,1000))
    worksheet[chr(66) + str(row)]= getReturn(ticker,1000)
    col+=1
    row +=1
print(ROI/20)
workbook.save('returns_bollinger.xlsx') 