# data_processor.py
import pandas as pd
from datetime import datetime

def process_sales_data(file_path):
    """Process sales data from CSV and return aggregated statistics."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    
    monthly_sales = df.groupby('month')['amount'].sum()
    return monthly_sales.to_dict()