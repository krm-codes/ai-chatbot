import streamlit as st
from datetime import datetime
import os
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from chatbot import load_excel_data, get_order_detail, parse_user_input

# Function to load the general chatbot
def get_general_chatbot():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# Function to get a response from the general chatbot with context
def get_general_chat_response(model, tokenizer, user_input, chat_history):
    chat_context = "\n".join(chat_history + [f"Human: {user_input}", "Bot:"])
    input_ids = tokenizer.encode(chat_context, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=150,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    ai_response = response.split("Bot:")[-1].strip()
    return ai_response

# Save chat history to a file
def save_chat_history(filename, chat_history):
    with open(filename, 'w') as f:
        for sender, message, timestamp in chat_history:
            f.write(f"{timestamp} - {sender}: {message}\n")

# Load chat history from a file
def load_chat_history(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        chat_history = []
        for line in lines:
            timestamp, message = line.split(' - ', 1)
            sender, message = message.split(': ', 1)
            chat_history.append((sender.strip(), message.strip(), timestamp.strip()))
        return chat_history
    return []


# -----------------------------------------------------------------------------------------------------
def display_chat_history():
    # Add an image at the top of the sidebar
    st.sidebar.image('bot.png', use_column_width=True)

    st.sidebar.title("Chat Sessions")

# ----------------------------------------------------------------------------------------------------
 # Start a new chat session (place this above chat history)
    if st.sidebar.button("Start New Chat"):
        st.session_state['chat_history'] = []
        st.session_state['selected_chat'] = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
# ------------------------------------------------------------------------------------------------------
    
    # Get chat history files
    history_files = [f for f in os.listdir('conversation') if f.endswith('.txt')]
    
    # Display 'Previous Chats' in black color
    st.sidebar.markdown(
        '<p style="color:black; font-weight:bold;">Previous Chats:</p>', 
        unsafe_allow_html=True
    )
    
    # Display chat history buttons
    for file in history_files:
        if st.sidebar.button(file.replace('.txt', '')):
            st.session_state['selected_chat'] = file
            st.session_state['chat_history'] = load_chat_history(f"conversation/{file}")



# Handle user input for general chat
def handle_general_chat_input():
    user_input = st.session_state.user_input
    if user_input:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state['chat_history'].append(("Human", user_input, timestamp))

        bot_response = get_general_chat_response(
            st.session_state['model'],
            st.session_state['tokenizer'],
            user_input, 
            [msg for _, msg, _ in st.session_state['chat_history'][-10:]]
        )
        st.session_state['chat_history'].append(("Bot", bot_response, timestamp))

        chat_filename = f"conversation/{st.session_state['selected_chat']}"
        save_chat_history(chat_filename, st.session_state['chat_history'])

        st.session_state.user_input = ""

# Handle user input for order-related queries
def handle_order_query_input():
    user_input = st.session_state.user_input
    if user_input:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state['chat_history'].append(("You", user_input, timestamp))

        # Load Excel Data
        orders_df, _, _ = load_excel_data('Orders.xlsx')

        # Column mappings for user queries
        column_mapping = {
            'quantity ordered': 'QUANTITYORDERED',
            'price': 'PRICEEACH',
            'order line number': 'ORDERLINENUMBER',
            'sales': 'SALES',
            'order date': 'ORDERDATE',
            'status': 'STATUS',
            'quarter id': 'QTR_ID',
            'month id': 'MONTH_ID',
            'year id': 'YEAR_ID',
            'product line': 'PRODUCTLINE',
            'msrp': 'MSRP',
            'product code': 'PRODUCTCODE',
            'customer name': 'CUSTOMERNAME',
            'phone': 'PHONE',
            'address line 1': 'ADDRESSLINE1',
            'address line 2': 'ADDRESSLINE2',
            'city': 'CITY',
            'state': 'STATE',
            'postal code': 'POSTALCODE',
            'country': 'COUNTRY',
            'territory': 'TERRITORY',
            'contact last name': 'CONTACTLASTNAME',
            'contact first name': 'CONTACTFIRSTNAME',
            'deal size': 'DEALSIZE'
        }

        # Parse user input to extract order number and intent
        order_number, column_name = parse_user_input(user_input, column_mapping)

        # Respond based on detected intent and order number
        if order_number and column_name:
            detail = get_order_detail(order_number, column_name, orders_df)
            if detail:
                bot_response = f"The {column_name} of order {order_number} is: {detail}."
            else:
                bot_response = f"Sorry, I couldn't find any order with number {order_number} or detail for {column_name}."
        else:
            bot_response = "Please provide a valid order number and specify a detail you're asking for (e.g., status, order date, etc.)."

        # Add bot response to chat history with timestamp
        st.session_state['chat_history'].append(("Bot", bot_response, timestamp))

        # Save chat history after each interaction
        chat_filename = f"conversation/{st.session_state['selected_chat']}"
        save_chat_history(chat_filename, st.session_state['chat_history'])

        # Clear the input field
        st.session_state.user_input = ""

# Function to add custom CSS
def add_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #2b2b2b;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #3a3a3a;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4a4a4a;
        color: #ffffff;
    }
    .stSidebar .stButton>button {
        width: 100%;
    }
    .chat-header {
        text-align: center;
        padding: 20px 0;
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
    }
    .stRadio > label {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to display the icon and chatbot name
def display_header(chatbot_name):
    icon_path = "bot.png"  # Replace with the path to your icon
    if os.path.exists(icon_path):
        with open(icon_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{encoded_string}" width="100">
                <div class="chat-header">{chatbot_name}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(f'<div class="chat-header">{chatbot_name}</div>', unsafe_allow_html=True)

# Main function for the Streamlit chatbot app
def main():
    st.set_page_config(page_title="ManviMate Advanced Response Assistant", layout="wide")

    # Add custom CSS for dark theme
    add_custom_css()

    # Display icon and chatbot name
    display_header("ManviMate Advanced Response Assistant")

    # Sidebar for chat history management
    display_chat_history()

    # Initialize chat session if not present
    if 'selected_chat' not in st.session_state or 'chat_history' not in st.session_state:
        st.session_state['selected_chat'] = "New Chat"
        st.session_state['chat_history'] = []

    # Add a radio button to switch between general chat and order query
    chat_mode = st.radio("Select Chat Mode:", ("General Chat", "Order Query"))

    # Chat display
    chat_container = st.container()
    with chat_container:
        for sender, message, timestamp in st.session_state['chat_history']:
            if sender in ["Human", "You"]:
                st.markdown(f"<div style='background-color: #3a3a3a; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>You ({timestamp}):</strong> {message}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color: #4a4a4a; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>Bot ({timestamp}):</strong> {message}</div>", unsafe_allow_html=True)

    # Input bar at the bottom for chatting
    st.markdown(
        """
        <style>
        [data-testid="stTextInput"] {
            position: fixed;
            bottom: 0;
            background-color: #2b2b2b;
            padding: 10px;
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Text input with a callback to handle user input and clear the field after submission
    if chat_mode == "General Chat":
        st.text_input("You: ", key="user_input", placeholder="Type your message here...", on_change=handle_general_chat_input, label_visibility="collapsed")
    else:
        st.text_input("You: ", key="user_input", placeholder="Ask anything related to your order...", on_change=handle_order_query_input, label_visibility="collapsed")

if __name__ == "__main__":
    # Ensure the 'conversation' directory exists
    if not os.path.exists('conversation'):
        os.makedirs('conversation')

    # Initialize chatbot for general chat
    if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
        st.session_state['model'], st.session_state['tokenizer'] = get_general_chatbot()

    main()