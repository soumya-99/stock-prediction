import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly

# import matplotlib.pyplot as mt

import plotly.graph_objs as go

st.sidebar.title("Stock Price Prediction...")
st.sidebar.image("./img/stock.png")
st.sidebar.markdown("""
## Enter the Stock Ticker Symbol and voila!
- You can choose the Start Date (Optional)
- Change the range of the prediction (Optional)
- Check the table to see the raw data
- Check the graph to see the visual representation of the data
- Finally check the forecast to see the prediction of the stock price
- You can change the theme with the button on the top right corner hamburger menu

[Follow on GitHub](https://github.com/soumya-99/)

[Check out Website](https://soumya-99.vercel.app/)
""")

TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Price Prediction")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "AMC", "TSLA", "BTC-USD", "ETH-USD", "DOGE-USD")
selected_stocks = st.selectbox("Select Your Favorite Stock", stocks)

START = st.date_input("Choose Start Date", value=date(2016, 1, 1))
# START = "2016-01-01"

n_years = st.slider("Years of Prediction:", 1, 4)
period_in_days = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Loading data...")
data = load_data(selected_stocks)
data_load_state.text("Data Loaded Successfully!")

st.subheader("Fetched Data")
st.write(data.tail())


def plot_data():
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=data["Date"],
    #               y=data["Open"], name="Stock Open"))
    # fig.add_trace(go.Scatter(x=data["Date"],
    #               y=data["Close"], name="Stock Close"))
    # fig.layout.update(title_text="Time Series Data",
    #                   xaxis_rangeslider_visible=True)
    # st.plotly_chart(fig)

    st.line_chart(data[['Open', 'Close']])


plot_data()
# plot_data(data[['Open', 'Close']])


# Forecasting
df_train = data[['Date', 'Close']]
df_train['Date'] = df_train['Date'].dt.tz_localize(
    None)  # Remove timezone info
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period_in_days)
forecast = model.predict(future)

st.subheader("Forecasted Data & Graph")
st.write(forecast.tail())

st.plotly_chart(plot_plotly(model, forecast))

st.write("Forecasted Components")
st.write(model.plot_components(forecast))
