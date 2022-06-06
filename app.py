from importlib import import_module
import pandas as pd
import numpy as np

import matplotlib as plot

from keras.models import load_model
import streamlit as st


start = '2010-01-01'
end= '2021-12-31'

st.title('Stocks Moving Average📉 ')


user_input = st.text_input('Enter Stock Ticker 💸', 'ITC.NS')
df = pdr.DataReader( user_input, 'yahoo', start, end)


#Describing Data

st.subheader('Data from 2010- 2021')
st.write(df.describe())

#Visualizations

st.subheader('Closing price vs Time chart📉 ')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close , 'b')
st.pyplot(fig)




st.subheader('Closing price vs Time Chart with 50MA📉 ')

ma50 =df.Close.rolling(50).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma50 )
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time Chart with 100MA📉 ')

ma100 =df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100 )
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time Chart with 200MA📉 ')

ma200 =df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time Chart with  50MA, 100MA & 200MA📉 ')
ma50 =df.Close.rolling(500).mean()
ma100 =df.Close.rolling(100).mean()
ma200 =df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma50, 'r')
plt.plot(ma100, 'g')
plt.plot(ma200, 'y')
plt.plot(df.Close)
st.pyplot(fig)


# data_tr =pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
# data_test =pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
# print(data_tr.shape)
# print(data_test.shape)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0,1))

# data_tr_array = scaler.fit_transform(data_tr)


# x_train =[]
# y_train =[]

# for i in range (100, data_tr_array.shape[0]):
#     x_train.append(data_tr_array[i-100: i])
#     y_train.append(data_tr_array[i: 0])

# x_train, y_train =np.array(x_train), np.array(y_train)

# model = load_model('keras_model.h5')

# past_100_days =data_tr.tail(100)
# final_df =past_100_days.append(data_tr, ignore_index=True)
# input_data = scaler.fit_transform(final_df)

# x_test = []
# y_test =[]
# import pandas as pd
# import numpy as np
# for i in range(100, input_data.shape[0]):
#     x_test.append(input_data[i-100: i])
#     y_test.append(input_data[i,0])


#     x_test, y_test = np.array(x_test), np.array(y_test)
#     y_predicted =model.predict(x_test)
#     scaler = scaler.scale_
#     scale_factor = 1/0.0245996
#     y_predicted = y_predicted * scale_factor 
#     y_test = y_test* scale_factor

# st.subheader('Predictions vs Original')
# fig2 = plt.figure(figsize=(12,6))
# plt.plot(y_test,  label ='Original Price')
# plt.plot(y_predicted, 'r', label ='Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# st.pyploy()


