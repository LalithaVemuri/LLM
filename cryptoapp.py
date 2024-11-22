import os
import requests
import streamlit as st
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import psycopg2  # For PostgreSQL database connection
import plotly.graph_objs as go
import pandas as pd

# Define Twelve Data API details
API_URL = "https://api.twelvedata.com/time_series"
API_KEY = "ad8ecb77b5da48cc9c912ccbf1b8fd41"  # Replace with your actual Twelve Data API key
SYMBOL = "AAPL"  # Example symbol, replace with any valid symbol
INTERVAL = "1day"  # Interval for the time series (1day, 1min, etc.)
START_DATE = "2024-11-5"
END_DATE = "2024-11-9"

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
                "AAPL",  # Symbol
                "1day",  # Interval
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

def format_data(data):
    """Format the API response into a readable string for LangChain."""
    formatted_data = "\n".join([f"{entry['datetime']}: Open={entry['open']}, Close={entry['close']}, High={entry['high']}, Low={entry['low']}" for entry in data])
    return formatted_data

def create_pdf(data):
    """Create a PDF file from the formatted data."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter  # Default page size (8.5 x 11 inches)
    
    # Set up PDF styles
    c.setFont("Helvetica", 10)
    
    # Add a title
    c.drawString(30, height - 40, "Time Series Data Report")
    
    # Add data to the PDF
    y_position = height - 60
    for line in data.split("\n"):
        if y_position <= 40:  # If space is running out, create a new page
            c.showPage()
            c.setFont("Helvetica", 10)
            y_position = height - 40
        c.drawString(30, y_position, line)
        y_position -= 12
    
    c.save()
    
    buffer.seek(0)  # Rewind the buffer to the beginning
    return buffer

def plot_stock_data(data):
    """Plot stock data with up/down movements on a graph."""
    # Convert data to a pandas DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Convert datetime to pandas datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Calculate the 'up' and 'down' movement
    df['price_change'] = df['close'].astype(float).diff()
    
    # Define color based on price change (green for up, red for down)
    df['color'] = df['price_change'].apply(lambda x: 'green' if x > 0 else 'red')

    # Create a plotly figure
    fig = go.Figure()

    # Add the candlestick plot
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'].astype(float),
        high=df['high'].astype(float),
        low=df['low'].astype(float),
        close=df['close'].astype(float),
        name='Candlestick'
    ))

    # Highlight up and down days with markers
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['close'].astype(float),
        mode='markers',
        marker=dict(color=df['color'], size=10),
        name='Price Movement'
    ))

    fig.update_layout(
        title=f"Stock Price Movement ({SYMBOL})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    
    # Show the plot in Streamlit
    st.plotly_chart(fig)

# Fetch and format data
data = fetch_time_series(START_DATE, END_DATE, SYMBOL, INTERVAL, API_KEY)

if data:
    time_series_data = format_data(data)

    # Insert data into PostgreSQL
    insert_data_to_db(data)

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
    result = chain.run(data=time_series_data)
    
    # Display the result in Streamlit
    st.write(f"### Insights based on time series data:\n{result}")
    
    # Create a PDF from the formatted data
    pdf_buffer = create_pdf(time_series_data)
    
    # Provide the download link
    st.download_button(
        label="Download PDF",
        data=pdf_buffer,
        file_name="time_series_report.pdf",
        mime="application/pdf"
    )

    # Plot the stock data
    plot_stock_data(data)

else:
    st.warning("No data found for the specified date range or symbol.")
