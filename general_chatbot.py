# from transformers import pipeline

# # Initialize the general chatbot
# def get_general_chatbot():
#     # Use 'text-generation' instead of 'conversational'
#     chatbot = pipeline('text-generation', model="microsoft/DialoGPT-medium", pad_token_id=50256)
#     return chatbot

# # Function to get a response from the general chatbot
# def get_general_chat_response(chatbot, user_input, chat_history):
#     # Combine user input with chat history
#     chat_context = "\n".join(chat_history + [f"You: {user_input}"])
    
#     # Generate response using the chatbot pipeline
#     response = chatbot(chat_context, max_length=150, num_return_sequences=1, truncation=True)

#     # Extract the generated text
#     generated_text = response[0]['generated_text']
    
#     # To avoid the bot repeating the user input, we can trim the input from the generated response.
#     # Find the last occurrence of "You: {user_input}" in the generated text and return the text after that.
#     if f"You: {user_input}" in generated_text:
#         # Get the index where the user input starts in the generated text
#         user_input_index = generated_text.rindex(f"You: {user_input}") + len(f"You: {user_input}")
#         generated_text = generated_text[user_input_index:].strip()
    
#     # Strip any leading context or user prompt
#     generated_text = generated_text.replace('Bot:', '').strip()
    
#     return generated_text

# # Example usage
# if __name__ == "__main__":
#     chatbot = get_general_chatbot()
#     chat_history = []  # Initialize chat history
    
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == 'exit':
#             break  # Exit the loop on 'exit'
#         chat_history.append(user_input)  # Append user input to chat history
#         response = get_general_chat_response(chatbot, user_input, chat_history)
#         chat_history.append(response)  # Append bot response to chat history
#         print("Bot:", response)
# -----------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_general_chatbot():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def get_general_chat_response(model, tokenizer, user_input, chat_history):
    # Combine chat history and new user input
    chat_context = "\n".join(chat_history + [f"Human: {user_input}", "AI:"])
    
    # Encode the input
    input_ids = tokenizer.encode(chat_context, return_tensors="pt")
    
    # Generate a response
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
    
    # Decode the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the AI's response
    ai_response = response.split("AI:")[-1].strip()
    
    return ai_response

if __name__ == "__main__":
    model, tokenizer = get_general_chatbot()
    chat_history = []
    
    print("Chatbot initialized. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        response = get_general_chat_response(model, tokenizer, user_input, chat_history)
        print("Bot:", response)
        
        chat_history.append(f"Human: {user_input}")
        chat_history.append(f"AI: {response}")
        
        # Keep only the last 5 turns to manage context length
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]