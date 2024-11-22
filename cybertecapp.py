import os
#import sys
import uuid
import requests
import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import psycopg2
import numpy as np
from uuid import UUID
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# Define Twelve Data API details
API_URL = "https://api.twelvedata.com/time_series"
API_KEY = "ad8ecb77b5da48cc9c912ccbf1b8fd41"  # Replace with your actual Twelve Data API key

# Streamlit inputs for user-defined dates, stock symbol, and interval
st.title("Stock Data Analysis: Up/Down Price Movements")
# Button to trigger the data fetching and analysis process
fetch_data_button = st.button("Fetch Data and Analyze")
SYMBOL = st.text_input("Enter the stock symbol (e.g., AAPL, MSFT, etc.):", "AAPL")
START_DATE = st.date_input("Start Date", pd.to_datetime("2020-04-06"))
END_DATE = st.date_input("End Date", pd.to_datetime("2020-04-20"))
INTERVAL = st.selectbox("Select the time interval for data retrieval:", ["1min", "5min", "15min", "30min", "1hour", "1day", "1week"], index=5)
PRICE_TYPES = st.multiselect("Select the price types to analyze:", ["open", "close", "high", "low"], default=["close"])
SHOW_INSIGHTS = st.selectbox("Would you like to see insights based on the data?", ["Yes", "No"], index=0)

# Validate UUID to ensure it's a string
def validate_uuid(run_id):
    """Ensure run_id is a string or UUID object."""
    if isinstance(run_id, UUID):
        return str(run_id)  # Convert UUID to string if it's a UUID object
    elif isinstance(run_id, str):
        return run_id  # It's already a string, so just return it
    else:
        raise ValueError("Invalid UUID or string format for run_id.")
    
# Database connection utility
def connect_to_db():
    """Establish a connection to the PostgreSQL database."""
    return psycopg2.connect(
        host="localhost",  # Change to your database host
        dbname="postgres",  # Change to your database name
        user="postgres",  # Change to your database user
        password="new_password"  # Change to your database password
    )
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


# Database operations
def check_data_exists(symbol, start_date, end_date, interval):
    """Check if data exists in both stock_data and t_timeseries tables."""
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cursor:
                # Check in stock_data table
                cursor.execute("""
                    SELECT COUNT(*) FROM stock_data
                    WHERE symbol = %s AND datetime >= %s AND datetime <= %s AND interval = %s
                """, (symbol, start_date, end_date, interval))
                stock_data_count = cursor.fetchone()[0]

                # Check in t_timeseries table
                cursor.execute("""
                    SELECT COUNT(*) FROM t_timeseries
                    WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s AND interval = %s
                """, (symbol, start_date, end_date, interval))
                timeseries_count = cursor.fetchone()[0]

                return stock_data_count > 0 and timeseries_count > 0
    except Exception as e:
        st.error(f"Error while checking data existence: {e}")
        return False

def insert_data_to_db(data, symbol, interval):
    """Insert time series data into both stock_data and t_timeseries tables."""
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cursor:
                # Insert data into stock_data table
                insert_stock_data_query = """
                INSERT INTO stock_data (datetime, open, high, low, close, volume, symbol, interval, currency, exchange, mic_code, type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (datetime, symbol, interval) DO NOTHING;  -- Avoid insertion if conflict occurs
                """
                for entry in data:
                    values = (
                        entry["datetime"], entry["open"], entry["high"], entry["low"],
                        entry["close"], entry["volume"], symbol, interval, "USD", "NASDAQ", "XNGS", "Common Stock"
                    )
                    cursor.execute(insert_stock_data_query, values)

                # Insert data into t_timeseries table
                insert_timeseries_query = """
                INSERT INTO t_timeseries (symbol, timestamp, data, interval)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (timestamp, symbol, interval) DO NOTHING;  -- Avoid insertion if conflict occurs
                """
                for entry in data:
                    values = (symbol, entry["datetime"], entry["close"], interval)
                    cursor.execute(insert_timeseries_query, values)

                conn.commit()
                st.success("Data inserted successfully into both PostgreSQL tables!")
    except Exception as e:
        st.error(f"Error while inserting data into PostgreSQL: {e}")



def fetch_data_from_db(symbol, start_date, end_date, interval):
    """Fetch stock data from the PostgreSQL database."""
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cursor:
                query = """
                SELECT datetime, open, high, low, close, volume FROM stock_data
                WHERE symbol = %s AND datetime >= %s AND datetime <= %s AND interval = %s
                ORDER BY datetime ASC
                """
                cursor.execute(query, (symbol, start_date, end_date, interval))
                data = cursor.fetchall()

                if data:
                    column_names = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                    return [dict(zip(column_names, row)) for row in data]
                return None
    except Exception as e:
        st.error(f"Error while fetching data from the database: {e}")
        return None


def check_if_view_exists():
    """Check if the view 'v' exists in the database."""
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.views
                        WHERE table_name = 'v'
                    );
                """)
                return cursor.fetchone()[0]
    except Exception as e:
        st.error(f"Error while checking if the view exists: {e}")
        return False
# Check if data is available in the database
data = fetch_data_from_db(SYMBOL, START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"), INTERVAL)

# Function to display insights based on the data
def display_insights(data):
    """Display insights based on the data."""
    prompt_template = """
    You are a financial analyst. Based on the following time series data, provide insights such as trends, patterns, and any significant observations.

    Data:
    {data}
    """
    
    # Set up LangChain components with OpenAI API key
    llm = OpenAI(temperature=0.7, openai_api_key="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")  # Replace with your actual OpenAI API key
    chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt_template, input_variables=["data"]))

    # Ensure run_id is passed as a string
    run_id = uuid.uuid4()  # Generate a UUID
    run_id = str(run_id)  # Convert the UUID to a string

    # Generate insights using LangChain
    result = chain.run(data=data)

    # Display the result in Streamlit
    st.write(f"### Insights based on time series data:\n{result}")

def create_view_v():
    """Create the view 'v' in the database if it doesn't already exist."""
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE VIEW v AS
                    SELECT  *, 
                        string_agg(CASE WHEN diff > 0 THEN 'u'::text ELSE 'd'::text END, '') 
                        OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) AS encoded
                    FROM    (
                        SELECT  *, 
                            data - lag(data, 1) OVER (ORDER BY id) AS diff
                        FROM    t_timeseries
                    ) AS x;
                """)
                conn.commit()
                st.success("View 'v' created successfully!")
    except Exception as e:
        st.error(f"Error while creating view 'v': {e}")

def fetch_encoded_data(symbol, start_date, end_date, interval):
    """Fetch encoded data from the view `v` in the database."""
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cursor:
                query = """
                    SELECT ts.timestamp, ts.data AS close,  -- Change ts.close to ts.data
                        string_agg(CASE WHEN diff > 0 THEN 'u'::text ELSE 'd'::text END, '')
                        OVER (ORDER BY ts.id ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) AS encoded
                    FROM t_timeseries ts
                    LEFT JOIN (
                        SELECT *, data - lag(data, 1) OVER (ORDER BY id) AS diff
                        FROM t_timeseries
                    ) AS diff_data
                    ON ts.id = diff_data.id
                    WHERE ts.symbol = %s AND ts.timestamp >= %s AND ts.timestamp <= %s AND ts.interval = %s
                    GROUP BY ts.timestamp, ts.data, ts.id, diff_data.diff  -- Add diff_data.diff to GROUP BY
                    ORDER BY ts.timestamp;
                """
                cursor.execute(query, (symbol, start_date, end_date, interval))
                data = cursor.fetchall()

                if data:
                    return [dict(zip(['timestamp', 'close', 'encoded'], row)) for row in data]
                return None
    except Exception as e:
        st.error(f"Error while fetching encoded data: {e}")
        return None

# Function to calculate the up/down movements based on the closing prices
def calculate_up_down_movements(data):
    """Calculate the up/down movements based on the closing prices."""
    movements = []
    for i in range(1, len(data)):
        prev_close = data[i-1]["close"]
        curr_close = data[i]["close"]
        if curr_close > prev_close:
            movements.append('u')  # 'u' for up
        else:
            movements.append('d')  # 'd' for down

    # Add 'no movement' for the first data point
    movements.insert(0, 'start')
    return movements


def plot_encoded_chart(data, movements):
    """Plot the up/down price movements."""
    times = [pd.to_datetime(entry["timestamp"]).strftime('%b %d') for entry in data] 

    close_prices = [entry["close"] for entry in data]

    # Convert encoded 'u' -> +1 (up) and 'd' -> -1 (down)
    movement_values = [1 if movement == 'u' else -1 for movement in movements]

    # Create the figure
    fig = go.Figure()

    # Plot the closing prices
    fig.add_trace(go.Scatter(
        x=times, 
        y=close_prices,  # Plot the closing prices
        mode='lines+markers',  # Line plot with markers
        line=dict(color='blue', width=2),
        name="Closing Price"
    ))

    # Plot up/down movement markers
    fig.add_trace(go.Scatter(
        x=times, 
        y=close_prices,  # Using close_prices for alignment on the y-axis
        mode='markers+text',
        text=movements,  # Display 'u' for up, 'd' for down as text markers
        textposition='top center',  # Position text above the markers
        marker=dict(
            color=['green' if movement == 'u' else 'red' for movement in movements],  # Green for up, red for down
            size=10,
            symbol='circle'  # Use circle markers for up/down movement (simple circle)
        ),
        name="Up/Down Movement"
    ))

    fig.update_layout(
        title=f"Stock Movements for {SYMBOL}",
        xaxis_title="Date",
        yaxis_title="Closing Price",
        template="plotly_dark",
        xaxis=dict(
            tickformat="%b%d",  # Month and Day (e.g., Nov-19)
            tickangle=360,  # Rotate the labels for better readability
            type='category'  # Treat the x-axis as categorical dates
        ),
        yaxis=dict(
            title="Closing Price",
            showgrid=True
        )
    )

    # Show the plot
    st.plotly_chart(fig)
    
    
 
# Generate and display insights if the user selects 'Yes' from the dropdown
if fetch_data_button: 
  
    # Check if the view 'v' exists, create it if necessary, and fetch the encoded data
    if not check_if_view_exists():
        create_view_v()

    # Fetch the encoded data from the view `v`
    encoded_data = fetch_encoded_data(SYMBOL, START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"), INTERVAL)

    # Plot the encoded chart if data is available
    if encoded_data:
        movements = calculate_up_down_movements(encoded_data)
        plot_encoded_chart(encoded_data,movements)
         # Show insights if the user selected 'Yes'
    if SHOW_INSIGHTS == "Yes":
            display_insights(data)
    else:
        st.warning("No data found for the selected symbol, date range, and interval.")
        
#else:
       # st.warning("No encoded data found in the database for the selected symbol, date range, and interval.")

    # Insert data if not available
if not check_data_exists(SYMBOL, START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"), INTERVAL):
        # Fetch the data (via API or other method) and insert into the DB
        # Placeholder for data retrieval code (e.g., API call)
        data_from_api = fetch_time_series(START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"), SYMBOL, INTERVAL, API_KEY)
        print(data_from_api)
        if data_from_api:
            insert_data_to_db(data_from_api, SYMBOL, INTERVAL)
