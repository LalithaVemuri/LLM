import os
import requests
import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import psycopg2  # For PostgreSQL database connection
from uuid import UUID
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Streamlit inputs for user-defined dates, stock symbol, and interval
st.title("Stock Data Analysis: Up/Down Price Movements")

SYMBOL = st.text_input("Enter the stock symbol (e.g., AAPL, MSFT, etc.):", "AAPL")
START_DATE = st.date_input("Start Date", pd.to_datetime("2024-11-05"))
END_DATE = st.date_input("End Date", pd.to_datetime("2024-11-09"))

# Set default interval to '1day'
INTERVAL = st.selectbox("Select the time interval for data retrieval:", ["1min", "5min", "15min", "30min", "1hour", "1day", "1week"], index=5)  # Default is '1day'

# Multiselect for price type to allow multiple selections
PRICE_TYPES = st.multiselect("Select the price types to analyze:", ["open", "close", "high", "low"], default=["close"])

# Add a dropdown for showing insights
SHOW_INSIGHTS = st.selectbox("Would you like to see insights based on the data?", ["No", "Yes"], index=0)

# Chart type selection
CHART_TYPE = st.selectbox("Select Chart Type:", 
                          ["Line Chart", "Candlestick Chart", "OHLC Chart", "Scatter Plot", "Area Chart", "Bar Chart", "Histogram", "Heatmap"], 
                          index=0)

# Database connection function
def connect_to_db():
    """Establish a connection to the PostgreSQL database."""
    conn = psycopg2.connect(
        host="localhost",  # Change to your database host
        dbname="postgres",  # Change to your database name
        user="postgres",  # Change to your database user
        password="new_password"  # Change to your database password
    )
    return conn

# Function to check if data exists in the database
def check_data_exists(symbol, start_date, end_date, interval):
    """Check if the data for the given symbol, date range, and interval exists in the database."""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        query = """
        SELECT COUNT(*) FROM stock_data
        WHERE symbol = %s AND datetime >= %s AND datetime <= %s AND interval = %s
        """
        cursor.execute(query, (symbol, start_date, end_date, interval))
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count > 0  # Return True if data exists, else False
    except Exception as e:
        st.error(f"Error while checking data existence: {e}")
        return False

# Function to insert data into the database
def insert_data_to_db(data, symbol, interval):
    """Insert time series data into the PostgreSQL database."""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO stock_data (datetime, open, high, low, close, volume, symbol, interval, currency, exchange, mic_code, type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        for entry in data:
            values = (
                entry["datetime"],
                entry["open"],
                entry["high"],
                entry["low"],
                entry["close"],
                entry["volume"],
                symbol,  # User-provided Symbol
                interval,  # User-selected interval
                "USD",  # Currency
                "NASDAQ",  # Exchange
                "XNGS",  # MIC code
                "Common Stock"  # Type
            )
            cursor.execute(insert_query, values)

        conn.commit()
        cursor.close()
        conn.close()
        st.success("Data inserted successfully into PostgreSQL database!")
    except Exception as e:
        st.error(f"Error while inserting data into PostgreSQL: {e}")

# Function to fetch data from the PostgreSQL database
def fetch_data_from_db(symbol, start_date, end_date, interval):
    """Fetch data from the PostgreSQL database."""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        query = """
        SELECT datetime, open, high, low, close, volume FROM stock_data
        WHERE symbol = %s AND datetime >= %s AND datetime <= %s AND interval = %s
        ORDER BY datetime ASC
        """
        cursor.execute(query, (symbol, start_date, end_date, interval))
        data = cursor.fetchall()

        if data:
            column_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            result = [dict(zip(column_names, row)) for row in data]
            cursor.close()
            conn.close()
            return result
        else:
            cursor.close()
            conn.close()
            return None
    except Exception as e:
        st.error(f"Error while fetching data from the database: {e}")
        return None

# Function to fetch time series data from Twelve Data API
def fetch_time_series(start_date, end_date, symbol, interval, api_key):
    """Fetch time series data from Twelve Data API."""
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key
    }
    
    response = requests.get("https://api.twelvedata.com/time_series", params=params)
    
    if response.status_code == 200:
        data = response.json()
        if "values" in data:
            return data['values']
        else:
            print("Error: No data found for the symbol.")
            return None
    else:
        st.write(f"API Request failed with status code: {response.status_code}")
        return None

# Function to plot stock data with selected chart type
def plot_stock_data(data, price_types, chart_type):
    """Plot stock data with up/down movements on a graph."""
    # Convert data to a pandas DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Convert datetime to pandas datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    if chart_type == "Line Chart":
        # Create a Plotly line chart
        fig = go.Figure()

        for price_type in price_types:
            df['price_change'] = df[price_type].astype(float).diff()
            df['color'] = df['price_change'].apply(lambda x: 'green' if x > 0 else 'red')
            
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df[price_type].astype(float),
                mode='lines+markers',
                marker=dict(color=df['color'], size=10),  # color markers based on price change
                line=dict(width=2),  # Line for the selected price
                name=f'{price_type.capitalize()} Price'
            ))

        fig.update_layout(
            title=f"Stock Price Movement ({SYMBOL})",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )

    elif chart_type == "Candlestick Chart":
        # Create a Plotly candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=f'{SYMBOL} Candlestick Chart'
        )])

        fig.update_layout(
            title=f"Stock Price Movement ({SYMBOL}) - Candlestick",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )

    # Show the plot in Streamlit
    st.plotly_chart(fig)

# Main logic: Check if data exists, fetch and insert if not found
data = fetch_data_from_db(SYMBOL, START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"), INTERVAL)

if not data:
    # If no data is found, insert new data from the API and then fetch it
    st.write("No data found in the database. Fetching new data...")
    new_data = fetch_time_series(START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"), SYMBOL, INTERVAL, "ad8ecb77b5da48cc9c912ccbf1b8fd41")
    
    if new_data:
        insert_data_to_db(new_data, SYMBOL, INTERVAL)
        # Re-fetch data after insertion
        data = fetch_data_from_db(SYMBOL, START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"), INTERVAL)

if data:
    # Plot the stock data with the selected price types and chart type
    plot_stock_data(data, PRICE_TYPES, CHART_TYPE)

    # Generate and display insights if the user selects 'Yes' from the dropdown
    if SHOW_INSIGHTS == "Yes":
        # Create a prompt template
        prompt_template = """
        You are a financial analyst. Based on the following time series data, provide insights such as trends, patterns, and any significant observations.

        Data:
        {data}
        """
        
        # Set up LangChain components with OpenAI API key
        llm = OpenAI(temperature=0.7, openai_api_key="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")  # Replace with actual OpenAI key
        chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt_template, input_variables=["data"]))
        
        # Generate insights using LangChain
        result = chain.run(data=data)

        # Display the result in Streamlit
        st.write(f"### Insights based on time series data:\n{result}")
else:
    st.warning("No data found in the database for the selected symbol, date range, and interval.")
