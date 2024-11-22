import os
import requests
import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import psycopg2  # For PostgreSQL database connection
import numpy as np  # For wave generation

# Streamlit inputs for user-defined dates, stock symbol, and interval
st.title("Stock Data Analysis: Up/Down Price Movements")

SYMBOL = st.text_input("Enter the stock symbol (e.g., AAPL, MSFT, etc.):", "AAPL")
START_DATE = st.date_input("Start Date", pd.to_datetime("2021-11-05"))
END_DATE = st.date_input("End Date", pd.to_datetime("2021-11-09"))

# Set default interval to '1day'
INTERVAL = st.selectbox("Select the time interval for data retrieval:", ["1min", "5min", "15min", "30min", "1hour", "1day", "1week"], index=5)  # Default is '1day'

# Multiselect for price type to allow multiple selections
PRICE_TYPES = st.multiselect("Select the price types to analyze:", ["open", "close", "high", "low"], default=["close"])

# Add a dropdown for showing insights
SHOW_INSIGHTS = st.selectbox("Would you like to see insights based on the data?", ["No", "Yes"], index=0)


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


# Plot the up/down movement chart
def plot_up_down_chart(data, movements):
    """Plot the up/down price movements."""
    times = [pd.to_datetime(entry["datetime"]) for entry in data]
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
            size=12,
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
            tickangle=45,  # Rotate the labels for better readability
            type='category'  # Treat the x-axis as categorical dates
        ),
        yaxis=dict(
            title="Closing Price",
            showgrid=True
        )
    )

    # Show the plot
    st.plotly_chart(fig)


# Main logic: Fetch data from the database and plot the up/down chart
data = fetch_data_from_db(SYMBOL, START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"), INTERVAL)

# If data is available, calculate movements and plot
if data:
    movements = calculate_up_down_movements(data)
    plot_up_down_chart(data, movements)
else:
    st.warning("No data found in the database for the selected symbol, date range, and interval.")
