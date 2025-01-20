import requests
import json

def call_llm(prompt: str, system_message: str = "Always answer in rhymes. Today is Thursday", model_name: str = "llama-3.2-3b-instruct", temperature: float = 0.7, max_tokens: int = -1, stream: bool = False):
    """
    Function to interact with the LLM through LMStudio's local API.

    Args:
    - prompt (str): The input prompt to the model.
    - system_message (str): The system message to set context for the model.
    - model_name (str): The model to use for generation.
    - temperature (float): The randomness of the output.
    - max_tokens (int): The maximum number of tokens to generate.
    - stream (bool): Whether to stream the response.

    Returns:
    - str: The response from the model.
    """

    url = "http://localhost:1234/v1/chat/completions"  # LMStudio API URL

    headers = {
        "Content-Type": "application/json",
    }

    # Construct the messages payload
    messages = [
        { "role": "system", "content": system_message },
        { "role": "user", "content": prompt }
    ]

    data = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream  # If True, it streams the response.
    }

    try:
        # Send POST request to the API
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # Check for successful response
        if response.status_code == 200:
            # Parse the response JSON
            response_data = response.json()
            
            # Check if 'choices' is in the response and contains the generated message
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                raise Exception("No valid response received from the model.")
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error calling the LLM: {e}")
        return None

if __name__ == '__main__':
    # Test the function with a prompt
    prompt = "What day is it today?"
    system_message = "Always answer in rhymes. Today is Thursday"  # Your custom system message
    model_name = "llama-3.2-3b-instruct"
    result = call_llm(prompt, system_message, model_name)

    if result:
        print("Model Response:", result)
    else:
        print("Failed to get a response from the model.")
