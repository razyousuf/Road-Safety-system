"""
#########################
#    generative_ai.py   #
#########################
This module integrates with Hugging Face's small Large Language Model (LLM) known as Phi-3-mini-4k-instruct to generate contextual road safety messages. 
It provides methods to:
- Send prompts to the LLM API for generating safety recommendations based on traffic, accident severity, and environmental data.
- Process and clean the generated text to avoid redundant or irrelevant information.
- Generate text and optional audio outputs (using Google Text-to-Speech) for accident warnings and safety alerts.

Main features:
- LLM-based text generation for safety recommendations.
- Contextual safety alerts based on live traffic, weather, and accident severity.
"""

import os
import json
import requests
import re
from gtts import gTTS
from io import BytesIO

# Ensure the Hugging Face API token is set
os.environ['HF_TOKEN'] = "Your_HuggingFace_Token" 

# Set up the Hugging Face Inference API endpoint and headers
HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"

HEADERS = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}"
}


 # ________________________________________________________________________________________________________________
def call_llm(prompt):
    """
    Function to call the Hugging Face large language model (LLM) and generate text based on the prompt.

    :param prompt: The input prompt for the model.
    :return: The main generated response from the model.
    """
    data = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 170},
        "task": "text-generation"
    }
    response = requests.post(HUGGING_FACE_API_URL, headers=HEADERS, json=data)
    
    if response.status_code != 200:
        raise Exception(f"Error calling LLM: {response.status_code} {response.text}")

    response_json = response.json()
    generated_text = response_json[0]["generated_text"]
    return generated_text


 # ________________________________________________________________________________________________________________
def generate_safety_message(data):
    severity_messages = {
        0: "Fatal accident area, be very careful!",
        1: "Serious accident area, be cautious!",
        2: "Slight accident area, stay alert!"
    }
    severity_message = severity_messages.get(data['predicted_severity'], "Unknown severity level.")

    # Base messages
    traffic_message = f"{data['current_traffic'].capitalize()} traffic ahead." #{data['Weather_Conditions']}."
    caution_message = ""  # Initialize the caution message to append to

    # Light Conditions Check
    if data['Light_Conditions'] == "Daylight":
        time_of_day_message = "Although it's daylight, remember that many accidents still occur under clear visibility conditions."
        caution_message += f" {time_of_day_message}"
    
    # Urban Area Check
    if data['Urban_or_Rural_Area'] == "Urban":
        urban_message = "Be extra cautious, especially in urban areas where most accidents happen."
        caution_message += f" {urban_message}"
    
    # Weather Conditions Check
    if data['Weather_Conditions'] == "Fine no high winds":
        weather_message = "Despite the fine weather, stay alert as most accidents happen under such conditions."
        caution_message += f" {weather_message}"

    # Friday and Time Check
    if data['Day_of_Week'] == "Friday":
        caution_message += " Stay extra cautious today as it's Friday, a high-risk period."

    current_hour = int(data['Time'].split(":")[0])
    if 16 <= current_hour <= 18:
        caution_message += " Pay special attention during this time, as accidents are more frequent around 5 PM."

    # Construct a more controlled prompt
    prompt = (
        f"{severity_message} {traffic_message} "
        f"{caution_message} "
        #"Provide any additional relevant safety information for this situation."
        "Provide relevant safety Recommendations: "
    )

    # Post-process to avoid redundancy and irrelevant content   
    generated_text = call_llm(prompt)

    # Remove unwanted characters and redundancy
    generated_text = generated_text.replace("Provide relevant safety", "").replace("### Response:\n", "").replace("\n\n", " ")
    unique_sentences = set()
    final_message = []
    for sentence in generated_text.split('.'):
        clean_sentence = sentence.strip()
        if clean_sentence and clean_sentence not in unique_sentences:
            unique_sentences.add(clean_sentence)
            final_message.append(clean_sentence)
        # Limit the response to the first 4-5 sentences to avoid lengthiness
        if len(final_message) >= 8:
            break

    result_message = '. '.join(final_message) + '.'
    result_message = re.sub(r'\s+\d+\.$', '', result_message)

    return result_message
