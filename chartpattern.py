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
import numpy as np

# Define Twelve Data API details
API_URL = "https://api.twelvedata.com/time_series"
API_KEY = "ad8ecb77b5da48cc9c912ccbf1b8fd41"  # Replace with your actual Twelve Data API key

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
SHOW_INSIGHTS = st.selectbox("Would you like to see insights based on the data?", ["No", "Yes"], index=0)  # Default is 'No'

# Add dropdown for chart patterns
CHART_PATTERN = st.selectbox("Select a chart pattern:", ["Line Chart", "Candlestick Chart", "Moving Average", "Volume Bar Chart", "Price Distribution Histogram", "RSI Chart"], index=0)

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

# Function to check if data already exists in the database
def check_data_exists(symbol, start_date, end_date, interval):
    """Check if the data for the given symbol, date range, and interval already exists in the database."""
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

        return count > 0  # Return True if data exists, False otherwise
    except Exception as e:
        st.error(f"Error while checking data existence: {e}")
        return False

# Function to insert data into PostgreSQL table
def insert_data_to_db(data):
    """Insert time series data into the PostgreSQL database."""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        # SQL Insert statement
        insert_query = """
        INSERT INTO stock_data (datetime, open, high, low, close, volume, symbol, interval, currency, exchange, mic_code, type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Extract the values from the JSON response and insert them
        for entry in data:
            values = (
                entry["datetime"],
                entry["open"],
                entry["high"],
                entry["low"],
                entry["close"],
                entry["volume"],
                SYMBOL,  # User-provided Symbol
                INTERVAL,  # User-selected interval
                "USD",  # Currency
                "NASDAQ",  # Exchange
                "XNGS",  # MIC code
                "Common Stock"  # Type
            )
            cursor.execute(insert_query, values)
        
        # Commit the transaction and close the connection
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

        # SQL query to fetch data for the given symbol, date range, and interval
        query = """
        SELECT datetime, open, high, low, close, volume FROM stock_data
        WHERE symbol = %s AND datetime >= %s AND datetime <= %s AND interval = %s
        ORDER BY datetime ASC
        """
        cursor.execute(query, (symbol, start_date, end_date, interval))
        data = cursor.fetchall()

        # If data exists, return it as a list of dictionaries
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
    
    response = requests.get(API_URL, params=params)
    
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

# Function to plot a line chart
def plot_line_chart(data, price_types):
    """Plot a line chart for the selected price types."""
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'])

    fig = go.Figure()

    for price_type in price_types:
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df[price_type].astype(float),
            mode='lines',
            name=f'{price_type.capitalize()} Price'
        ))

    fig.update_layout(
        title=f"Line Chart for {SYMBOL}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark"
    )

    st.plotly_chart(fig)

# Function to plot a candlestick chart
def plot_candlestick_chart(data):
    """Plot a candlestick chart."""
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'])

    fig = go.Figure(data=[go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlestick'
    )])

    fig.update_layout(
        title=f"Candlestick Chart for {SYMBOL}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark"
    )

    st.plotly_chart(fig)

# Function to plot a moving average chart
def plot_moving_average_chart(data, window=20):
    """Plot a moving average chart."""
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['moving_avg'] = df['close'].astype(float).rolling(window=window).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['close'].astype(float),
        mode='lines',
        name='Close Price'
    ))

    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['moving_avg'],
        mode='lines',
        name=f'{window}-Day Moving Average',
        line=dict(dash='dot')
    ))

    fig.update_layout(
        title=f"Moving Average Chart for {SYMBOL}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark"
    )

    st.plotly_chart(fig)

# Function to plot volume bar chart
def plot_volume_chart(data):
    """Plot a bar chart for stock volume."""
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['datetime'],
        y=df['volume'].astype(float),
        name='Volume',
        marker=dict(color='rgba(55, 83, 109, 0.7)')
    ))

    fig.update_layout(
        title=f"Volume Bar Chart for {SYMBOL}",
        xaxis_title="Date",
        yaxis_title="Volume",
        template="plotly_dark"
    )

    st.plotly_chart(fig)

# Function to plot a price distribution histogram
def plot_price_histogram(data):
    """Plot a histogram of closing prices."""
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'])

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df['close'].astype(float),
        name='Price Distribution',
        marker=dict(color='rgba(58, 71, 80, 0.7)')
    ))

    fig.update_layout(
        title=f"Price Distribution Histogram for {SYMBOL}",
        xaxis_title="Price (USD)",
        yaxis_title="Frequency",
        template="plotly_dark"
    )

    st.plotly_chart(fig)

# Function to plot RSI (Relative Strength Index)
def plot_rsi_chart(data, window=14):
    """Plot the RSI chart."""
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['close'] = df['close'].astype(float)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    df['rsi'] = rsi

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['rsi'],
        mode='lines',
        name=f'RSI ({window} periods)',
        line=dict(color='purple')
    ))

    fig.update_layout(
        title=f"RSI Chart for {SYMBOL}",
        xaxis_title="Date",
        yaxis_title="RSI",
        template="plotly_dark"
    )

    st.plotly_chart(fig)

# Ensure UUID is passed as a string for run_id
def validate_uuid(run_id):
    """Ensure run_id is a string or UUID object."""
    if isinstance(run_id, UUID):
        return str(run_id)  # Convert UUID to string
    elif isinstance(run_id, str):
        return run_id  # Already a string
    else:
        raise ValueError("Invalid UUID or string format for run_id.")

# Main logic to fetch, insert, and plot charts
if not check_data_exists(SYMBOL, START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"), INTERVAL):
    # If no data exists, fetch data from the API and insert into the database
    api_data = fetch_time_series(START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"), SYMBOL, INTERVAL, API_KEY)
    if api_data:
        insert_data_to_db(api_data)
else:
    # If data exists, fetch from the database
    data = fetch_data_from_db(SYMBOL, START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"), INTERVAL)

    if data:
        # Based on chart pattern selection, plot the corresponding chart
        if CHART_PATTERN == "Line Chart":
            plot_line_chart(data, PRICE_TYPES)
        elif CHART_PATTERN == "Candlestick Chart":
            plot_candlestick_chart(data)
        elif CHART_PATTERN == "Moving Average":
            plot_moving_average_chart(data)
        elif CHART_PATTERN == "Volume Bar Chart":
            plot_volume_chart(data)
        elif CHART_PATTERN == "Price Distribution Histogram":
            plot_price_histogram(data)
        elif CHART_PATTERN == "RSI Chart":
            plot_rsi_chart(data)

        # Generate and display insights if the user selects 'Yes' from the dropdown
        if SHOW_INSIGHTS == "Yes":
            # Create a prompt template
            prompt_template = """
            You are a financial analyst. Based on the following time series data, provide insights such as trends, patterns, and any significant observations.

            Data:
            {data}
            """
            
            # Create and pass the UUID as string to LangChain
            run_id = UUID('860b9a41-d572-4711-961e-bc9af872049b')  # Example UUID
            run_id_str = validate_uuid(run_id)

            # Set up LangChain components with OpenAI API key
            llm = OpenAI(temperature=0.7, openai_api_key="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")  # Replace with actual OpenAI key
            chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt_template, input_variables=["data"]))
            
            # Generate insights using LangChain
            result = chain.run(data=data)
            
            # Display the result in Streamlit
            st.write(f"### Insights based on time series data:\n{result}")
    else:
        st.warning("No data found. Please try a different symbol or date range.")
