import pandas as pd
import numpy as np
import openpyxl
import time 

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

from datetime import datetime , timedelta
import datetime as dt
import yfinance as yf

tickers =pd.read_excel("Porfolio.xlsx")['Tickers']
workbook = openpyxl.Workbook()


worksheet = workbook.active
start_time = time.time()


def calc_macd(data, len1,len2,len3):
    shortEMA = data.ewm(span = len1, adjust= False).mean()
    longEMA = data.ewm(span = len2, adjust= False).mean()
    MACD = shortEMA-longEMA
    signal = MACD.ewm(span = len3 , adjust= False).mean()
    return MACD , signal

def calc_rsi(data,period):
    delta = data.diff() #egy nappal shiftelt diff
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com = period, adjust=False).mean()
    ema_down = down.ewm(com = period, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100/(1+rs))
    return rsi

def calc_bollinger(data,period):
    mean = data.rolling(period).mean()
    std = data.rolling(period).std()
    upper_band = np.array(mean) + 2 * np.array(std)  
    lower_band = np.array(mean) - 2 * np.array(std)  
    return upper_band,lower_band

returns = 0
returns2 = 0
returns3 = 0
returns4 = 0
row = 1
Col = 65
worksheet["C" + str(row)] = "Lin_reg"
worksheet["D" + str(row)] = "Random_forest"
worksheet["E" + str(row)] = "SVR"
worksheet["F" + str(row)] = "ModLinKomb"
for ticker in tickers [0:1]:
    worksheet[chr(Col)+chr(Col)+ str(1)] =   ticker + "Lin_reg"
    worksheet[chr(Col)+chr(Col+1) + str(1)] = ticker + "Random_forest"
    worksheet[chr(Col)+ chr(Col+2) + str(1) ] = ticker + "SVR"
    worksheet[chr(Col)+ chr(Col+3) + str(1) ] = ticker + "Mod_lin_komb"
    worksheet["B" + str(row+1)] = ticker
    
    
    ROI = 1000
    owned = 0 
    ROI2 = 1000
    owned2 = 0 
    ROI3 = 1000
    owned3 = 0 
    ROI4 = 1000
    owned4 = 0
    end_date = datetime(2022,1,1)
    start_date = end_date - timedelta(days= 10 * 365)
    
    df = yf.download(ticker,start=start_date , end=end_date , interval='1d',prepost="false")
    #print(df.head())
    df = df.loc[:,["Open","Close","Volume","High","Low"]]
    
    df['Prev_close'] = df.loc[:,'Close'].shift(1)
    df['Prev_volume'] = df.loc[:,'Volume'].shift(1)
    df['Prev_Open']= df.loc[:,'Open'].shift(1)
    df['Prev_Hi']=df.loc[:,'High'].shift(1)
    df['Prev_lo']=df.loc[:,'Low'].shift(1)
    #print(df.head())
    
    
    datetimes = df.index.values
    weekdays = []
   
    for dt in datetimes:
        dt = datetime.strptime(str(dt), '%Y-%m-%dT%H:%M:%S.000000000')
        weekdays.append(dt.weekday())
    df['Weekday'] = weekdays    
    
    df["5SMA"] = df['Prev_close'].rolling(5).mean()
    df["10SMA"] = df['Prev_close'].rolling(10).mean()
    df["20SMA"] = df['Prev_close'].rolling(20).mean()
    df["50SMA"] = df['Prev_close'].rolling(50).mean()
    df["100SMA"] = df['Prev_close'].rolling(100).mean()
    df["200SMA"] = df['Prev_close'].rolling(200).mean()
    df["Move_direct"]= (1-df['Prev_Open'] / df["Prev_close"] )*100
    df["OBV"]=np.where(df['Prev_close'] > df['Prev_close'].shift(1), df['Prev_volume'], np.where(df['Prev_close'] < df['Prev_close'].shift(1), -df['Prev_volume'], 0)).cumsum()
    df["TR"]=np.maximum(df["Prev_Hi"]-df["Prev_lo"],df["Prev_Hi"]-df["Prev_close"].shift(1),df["Prev_close"].shift(1)-df["Prev_lo"])
    df['ATR14'] = df["TR"].rolling(14).mean()
    df["+DM"]=df["Prev_Hi"].shift(1)-df["Prev_Hi"]
    df["-DM"]=df["Prev_lo"].shift(1)-df["Prev_lo"]
    df["EMA14+"]=df["+DM"].ewm(com=0.1).mean()
    df["EMA14-"]=df["-DM"].ewm(com=0.1).mean()
    df["Prediction"]= df['Close'].transform(lambda x : np.sign(x.diff()))
    
    df["+DI14"]=(df["EMA14+"]/df['ATR14'])*100
    df["-DI14"]=(df["EMA14-"]/df['ATR14'])*100
    df["DI14"]= np.abs(df["+DI14"]-df["-DI14"]) / np.abs(df["+DI14"] + df["-DI14"])
    df["ADX14"]= (df["DI14"].shift(1)*13 + df["DI14"])*100
    df["ADXUT"]= np.where((df["ADX14"] < 25) & (df["ADX14"].shift(1) > 25) & (df["+DI14"] > df["-DI14"]),1,0)
    df["ADXDT"]= np.where((df["ADX14"] < 25) & (df["ADX14"].shift(1) > 25) & (df["+DI14"] < df["-DI14"]),-1,0)
    df["StcOsc"]= 100*(df["Prev_close"]-df["Prev_close"].rolling(14).min())/(df["Prev_close"].rolling(14).max() - df["Prev_close"].rolling(14).min())
    #df["Target"] = df.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Prev_close"]
    #print[df.head()]
    #ADL ADL ALD!!!!!!!!! kiszamitása
    #plt.plot(df["ADXUT"] ,color="Green")
    #plt.plot(df["ADXDT"] ,color="RED")
    #plt.plot(df["ADXDT"] ,color="Red")
    #
    #plt.show()
    #plt.plot(df["Prev_close"], color="Blue")
    #plt.show()
    
    macd,signal = calc_macd(df['Prev_close'],12,26,9)
    df['MACD'] = macd
    df['MACD_signal']=signal
    
    df['RSI'] = calc_rsi(df['Prev_close'],13)
    df['RSI_volume'] = calc_rsi(df['Prev_volume'],13)
    
    
    upper,lower = calc_bollinger(df['Prev_close'],20)
    df['upperBand'] = upper
    df['lowerBand'] = lower
    #print(df.tail())
    labels = ["Prev_close","Prev_volume","5SMA" ,"10SMA","20SMA","50SMA","100SMA" ,"200SMA" ,]
    
    period = 1
    new_labels = [str(period)+'d_'+label for label in labels]
    df[new_labels] = df[labels].pct_change(period, fill_method='ffill')
    
    period = 5
    new_labels = [str(period)+'d_'+label for label in labels]
    df[new_labels] = df[labels].pct_change(period, fill_method='ffill')
    
    labels.append("OBV" "ADX14"  "StcOsc" "Move_direct"  "MACD"  "MACD_signal" "RSI" "RSI_volume"  "ADXUT" "ADXD" )
    df = df.replace(np.inf , np.nan).dropna()
    #print(df.head(),df.tail())
    
    testcycle = 365
    
    
    for i in range(0,testcycle-1,1):
        
        Y = df["Close"]
        Y_RF = df["Prediction"]
        
        #df2 = df.drop(['Close','Volume'], axis = 1,)
        
        X=df[['StcOsc','Prev_close','Prev_volume','Prev_lo','Prev_Hi' , "MACD_signal","Move_direct","ADXUT","ADXDT","RSI","DI14", "OBV" , "5SMA" ,'MACD' ]]
        
        #correláció vizsgálat
        #corr = (X.corr())
        #print(corr)
        X_train = X[:-testcycle+i]
        Y_train = Y[:-testcycle+i]
        Y_train_rf = Y_RF[:-testcycle+i]
        X_test = X[i-testcycle:]
        scaler = MinMaxScaler( feature_range=(-1,1))
        X_train_rf = scaler.fit_transform(X_train)
        X_test_rf = scaler.fit_transform(X_test)
        
        
        model = LinearRegression()
        model2 = RandomForestClassifier(n_estimators=75 , oob_score=True , criterion="gini",random_state=0)
        model3 = SVR(kernel= 'poly' , degree= 4 , )
        
        
       
        model = model.fit(X_train,Y_train)
        model2 = model2.fit(X_train_rf,Y_train_rf)
        model3 = model3.fit(X_train,Y_train)
        
        
        
        tipp = model.predict(X_test)[0]
        tipp2 = model2.predict(X_test_rf)[0]
        tipp3 = model3.predict(X_test)[0]
        
        if tipp < df['Prev_close'].iloc[i-testcycle] :
            lrpredict = -1
        elif tipp> df['Prev_close'].iloc[i-testcycle]:
            lrpredict = +1    
        if tipp3 < df['Prev_close'].iloc[i-testcycle] :
            SVRpredict = -1
        elif tipp3 > df['Prev_close'].iloc[i-testcycle]:
            SVRpredict = +1    
        if tipp < df['Prev_close'].iloc[i-testcycle] and owned!=0 :
           ROI = owned * df['Prev_close'].iloc[i-testcycle]
           owned = 0
        elif tipp >  df['Prev_close'].iloc[i-testcycle] and owned == 0: 
           owned = ROI / float(df['Prev_close'].iloc[i-testcycle]) 
           ROI = 0 
        if tipp3 < df['Prev_close'].iloc[i-testcycle] and owned3!=0 :
           ROI3 = owned3 * df['Prev_close'].iloc[i-testcycle]
           owned3 = 0
        elif tipp3 >  df['Prev_close'].iloc[i-testcycle] and owned3 == 0: 
           owned3 = ROI3 / float(df['Prev_close'].iloc[i-testcycle]) 
           ROI3 = 0  
             
        if tipp2 == -1 and owned2!=0 :
           ROI2 = owned2 * df['Prev_close'].iloc[i-testcycle]
           owned2 = 0
        elif tipp2 ==1   and owned2 == 0:  
           owned2 = ROI2 / float(df['Prev_close'].iloc[i-testcycle]) 
           ROI2 = 0 
        if lrpredict * 0.4 + SVRpredict * 0.1 + tipp2 * 0.5 > 0 and owned4 == 0:
            owned4 = ROI4 / float(df['Prev_close'].iloc[i-testcycle])
            ROI4 = 0
        elif lrpredict * 0.4 + SVRpredict * 0.1 + tipp2 * 0.5 < 0 and owned4 !=0 :
            ROI4 = owned4 * df['Prev_close'].iloc[i-testcycle]
            owned4 = 0
         
        worksheet[chr(Col)+chr(Col)+ str(i+2)] =  max(ROI,owned*df['Prev_close'].iloc[i-testcycle]) 
        worksheet[chr(Col)+chr(Col+1) + str(i+2)] = max(ROI2,owned2*df['Prev_close'].iloc[i-testcycle])
        worksheet[chr(Col)+ chr(Col+2) + str(i+2) ] = max(ROI3,owned3*df['Prev_close'].iloc[i-testcycle])
        worksheet[chr(Col)+ chr(Col+3) + str(i+2) ] = max(ROI4,owned4*df['Prev_close'].iloc[i-testcycle])
        
    #print(ROI2,owned2)
    #print(max(ROI,owned*df['Close'].iloc[-1]) , "lin reg")
    #print(max(ROI2,owned2*df['Close'].iloc[-1]), "R_F")
    #print(max(ROI3,owned3*df['Close'].iloc[-1]) , "SVR")
    #print(ticker)
    worksheet['C'+ str(row+1)] = max(ROI,owned*df['Prev_close'].iloc[-1]) 
    worksheet['D'+ str(row+1)] = max(ROI2,owned2*df['Prev_close'].iloc[-1])
    worksheet['E'+ str(row+1) ] = max(ROI3,owned3*df['Prev_close'].iloc[-1])
    worksheet['F'+ str(row+1) ] = max(ROI4,owned4*df['Prev_close'].iloc[-1])
    returns += max(ROI,owned*df['Close'].iloc[-1])
    returns2 += max(ROI2,owned2*df['Close'].iloc[-1])
    returns3 += max(ROI3,owned3*df['Close'].iloc[-1])
    returns4 += max(ROI4,owned4*df['Close'].iloc[-1])
    #print ( (1000 / df['Prev_close'].iloc[-testcycle]) * df['Prev_close'].iloc[-1])
    #returns.append(ticker)       
    #returns.append(max(ROI,owned*df['Prev_close'].iloc[-1]))    
    #returns.append((1000 / df['Prev_close'].iloc[-testcycle]) * df['Prev_close'].iloc[-1])        
    #
    row +=1 
    Col +=1
workbook.save('returns_normalised_ver2.xlsx') 
print(returns/20 , returns2/20 , returns3/20 ,returns4/20)
end_time = time.time()
print(f"Runtime: {end_time - start_time} seconds")