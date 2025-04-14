from datetime import datetime

def format_and_print_user_input(user_input):
    # Get the current timestamp
    current_time = datetime.now()

    # Format the timestamp as a string
    timestamp_str = current_time.strftime("[%I:%M %p]")

    # Combine timestamp and user input
    formatted_message = f"\n{timestamp_str} User \n           {user_input}\n"

    print(formatted_message)

def print_log(log):
    # Get the current timestamp
    current_time = datetime.now()

    # Format the timestamp as a string
    timestamp_str = current_time.strftime("[%I:%M %p]")

    # Combine timestamp and user input
    formatted_message = f"\n{timestamp_str} Application Log \n           {log}\n"

    print(formatted_message) 

def format_and_print_genai_response(response):
    # Get the current timestamp
    current_time = datetime.now()

    # Format the timestamp as a string
    timestamp_str = current_time.strftime("[%I:%M %p]")

    # Combine timestamp and user input
    formatted_message = f"\n{timestamp_str} GenAI \n           {response}\n"

    print(formatted_message)