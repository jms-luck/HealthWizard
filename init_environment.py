import os
from dotenv import load_dotenv
import sys

def init_environment():
    """Initialize the environment variables and check dependencies."""
    # Load environment variables
    if not load_dotenv():
        print("Warning: .env file not found. Creating one with default values...")
        create_default_env()
    
    # Check for required environment variables
    required_vars = [
        'TAVILY_API_KEY',
        'GOOGLE_API_KEY',
        'SERPER_API_KEY',
        'AZURE_MAPS_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("Error: The following environment variables are missing:")
        for var in missing_vars:
            print(f"- {var}")
        sys.exit(1)

def create_default_env():
    """Create a default .env file with placeholder values."""
    env_content = """
TAVILY_API_KEY=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
GOOGLE_API_KEY=AIzaSyAO-n2dhhke4Cq_Iix1-bILoU6EY7VNsnM
SERPER_API_KEY=ed4acec1529a6f8755a04900d2554b5252aba850b59262e44712c7596509ef4a
AZURE_MAPS_KEY=EumXcWSYqKLcsw9zymB1cPRIfDzNbZBXO7BCjKsbsAITXSpRIZbMJQQJ99BBACYeBjFPDDZUAAAgAZMP1DsH
    """.strip()
    
    with open('.env', 'w') as f:
        f.write(env_content)

if __name__ == "__main__":
    init_environment()
