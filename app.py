import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.tools import DuckDuckGoSearchRun  # Updated import

# Load environment variables
_ = load_dotenv(find_dotenv())

# API Keys with proper error handling
def get_api_key(key_name, default_value):
    value = os.getenv(key_name)
    if not value:
        st.warning(f"{key_name} not found in environment variables. Using default value.")
        return default_value
    return value

TAVILY_API_KEY = get_api_key('TAVILY_API_KEY', "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")
GOOGLE_API_KEY = get_api_key('GOOGLE_API_KEY', "AIzaSyAO-n2dhhke4Cq_Iix1-bILoU6EY7VNsnM")
SERPER_API_KEY = get_api_key('SERPER_API_KEY', "ed4acec1529a6f8755a04900d2554b5252aba850b59262e44712c7596509ef4a")
AZURE_MAPS_KEY = get_api_key('AZURE_MAPS_KEY', "EumXcWSYqKLcsw9zymB1cPRIfDzNbZBXO7BCjKsbsAITXSpRIZbMJQQJ99BBACYeBjFPDDZUAAAgAZMP1DsH")

# Configure Google API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ...rest of existing code...
