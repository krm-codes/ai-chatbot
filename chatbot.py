import pandas as pd
import re
import os
from datetime import datetime
from transformers import pipeline

# Load Excel data
def load_excel_data(file):
    xls = pd.ExcelFile(file)
    orders_df = pd.read_excel(xls, 'orders')
    shipment_df = pd.read_excel(xls, 'shipment')
    invoice_df = pd.read_excel(xls, 'invoice')
    return orders_df, shipment_df, invoice_df

# Search order details from orders sheet
def get_order_detail(order_number, column_name, orders_df):
    result = orders_df[orders_df['ORDERNUMBER'] == order_number]
    
    # Check if the order exists
    if result.empty:
        return f"No order found for order number {order_number}."
    
    # Check if the column exists in the result
    if column_name in result.columns:
        detail_value = result[column_name].values[0]
        
        # Check if the detail value is missing or NaN
        if pd.isna(detail_value):
            return f"No data found for '{column_name}' in order {order_number}."
        
        return detail_value
    
    return f"Column '{column_name}' not found in the dataset."

# Initialize the chatbot
def get_chatbot():
    chatbot = pipeline('conversational', model="microsoft/DialoGPT-medium")
    return chatbot

# Function to parse order number and intent (status, shipment, or invoice)
def parse_user_input(user_input, column_mapping):
    # Regular expression to find a numeric order number in the user input
    order_number_match = re.search(r'\b\d{4,}\b', user_input)  # Order number with at least 4 digits
    order_number = int(order_number_match.group()) if order_number_match else None

    # Search for intent by matching words in input with columns in orders_df
    for keyword, column_name in column_mapping.items():
        if keyword in user_input.lower():
            return order_number, column_name
    return order_number, None

# Function to save chat history to a file
def save_chat_history(filename, chat_history):
    with open(filename, 'w') as f:
        for sender, message, timestamp in chat_history:
            f.write(f"{timestamp} - {sender}: {message}\n")

# Function to load chat history from a file
def load_chat_history(filename):
    chat_history = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                timestamp, sender_message = line.strip().split(' - ', 1)
                sender, message = sender_message.split(': ', 1)
                chat_history.append((sender, message, timestamp))
    return chat_history
