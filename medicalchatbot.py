#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import re
import sys
import time
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
from PIL import Image
import speech_recognition as sr

# LangChain 
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, llms
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult, SystemMessage, HumanMessage, AIMessage
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.llms.base import BaseLLM

# Translator for multilingual 
from googletrans import Translator
from typing import Optional, List


# In[9]:


load_dotenv()
GOOGLE_API_KEY = "AIzaSyCgaz3OFtXuNx-SCRPz2N58UCfpo0pcH_g"
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY is not set.")
genai.configure(api_key=GOOGLE_API_KEY)

translator = Translator()


# In[10]:


# Initialize translator
translator = Translator()

# Regular expression for parsing actions
ACTION_RE = re.compile(r'^Action:\s*(\w+)\s*:\s*(.+)$')


# In[11]:


class GoogleGenAI(LLM):
    @property
    def _llm_type(self) -> str:
        return "google-generativeai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self._call(prompt, stop)


# In[12]:


# Regular expression for parsing actions
action_re = re.compile(r'^Action:\s*(\w+)\s*:\s*(.+)$')

# Chatbot class for text-only multi-turn interaction
class Chatbot:
    def __init__(self, system):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result


    def execute(self):
        # Prepare prompt from the conversation history
        prompt = "\n".join([f'{msg["role"]}:{msg["content"]}' for msg in self.messages])
        model = genai.GenerativeModel("gemini-1.5-flash")
        raw_response = model.generate_content(prompt)
        # Extract text content from the first candidate
        result_text = raw_response.candidates[0].content.parts[0].text
        return result_text


# In[13]:


google_llm = GoogleGenAI()


# In[14]:


system_prompt = """
You are an expert medical doctor with extensive experience in diagnosing and treating patient queries. Your role is to engage patients in a structured, empathetic, and conversational flow to identify the root cause of their health issues—such as illnesses, aches, or discomforts—and provide tailored solutions. These solutions should include appropriate remedies, medication suggestions (with general disclaimers if needed), lifestyle advice, diet plans, and exercise recommendations to promote overall health and recovery.

For example:

Patient: "I’ve been having a fever."
Doctor: "How long have you been experiencing this fever?"
Patient: "For three days."
Doctor: "Are you also experiencing symptoms like a dry cough, runny nose, or body aches?"
Patient: "No, doctor."
Doctor: "Alright, let’s narrow this down. Have you noticed any other symptoms, like chills, fatigue, or loss of appetite? Also, have you been able to measure your temperature?"
Your responses should:

Ask follow-up questions to gather more details about symptoms, duration, severity, and any relevant medical history.
Analyze the information provided to suggest possible causes (e.g., viral infection, bacterial issue, etc.).
Recommend remedies (e.g., rest, hydration), over-the-counter medications if appropriate (e.g., paracetamol for fever), and caution about consulting a healthcare provider for prescriptions or severe cases.
Provide a simple, practical diet plan to support recovery (e.g., light meals, hydration tips) and address irregular eating habits for long-term health.
Suggest basic exercises or wellness tips (e.g., light stretching or breathing exercises) suitable for their condition.
Maintain a professional yet caring tone, ensuring the patient feels heard and supported.
If information is missing, politely prompt the patient to clarify. Avoid making definitive diagnoses requiring in-person tests, and include disclaimers like ‘Please consult a local doctor if symptoms persist or worsen.’ Ensure all advice aligns with general medical knowledge and promotes a healthy lifestyle.""".strip()

# Action functions using the Gemini API
def generate_workout(level):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Generate a workout plan for a {level} fitness level")
    return response.candidates[0].content.parts[0].text 

def suggest_meal(preferences):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Suggest a meal plan with {preferences}")
    return response.candidates[0].content.parts[0].text


# Mapping actions to functions
known_actions = {
    "generate_workout": generate_workout,
    "suggest_meal": suggest_meal,
    
}

# Function for text-based multi-turn conversation
def query(question, max_turns=5):
    i = 0
    bot = Chatbot(system_prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print("\nAssistant response:")
        print(result)
        # Look for an action command in the response
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}: {action_input}")
            print(f"\n-- running action: {action} with input: {action_input}")
            observation = known_actions[action](action_input.strip())
            print(f"\nObservation: {observation}\n")
            next_prompt = f"Answer: {observation}"
            # Pause briefly before the next turn
            time.sleep(1)
        else:
            # No further action, so end conversation
            return

# Function to capture voice input from the microphone and return recognized text
def get_voice_query():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Please speak your query now...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        voice_text = recognizer.recognize_google(audio)
        print(f"Recognized voice query: {voice_text}")
        return voice_text
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from the speech recognition service; {e}")
    return None

# Function to handle image queries.
# The user provides a path to an image and an optional accompanying text query.
def handle_image_query():
    image_path = input("Enter the path to the image file: ").strip()
    if not os.path.isfile(image_path):
        print("File not found. Please check the path and try again.")
        return
    try:
        image = Image.open(image_path)
        image.show()  # This will open the image using the default image viewer
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    user_query = input("Enter your query related to this image: ").strip()
    # Upload the file to Google Generative AI (if supported)
    try:
        sample_file = genai.upload_file(path=image_path, display_name="Image")
    except Exception as e:
        print(f"Error uploading image: {e}")
        return

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    # Create a prompt that includes both the query and the reference to the uploaded image
    prompt = (f"Using the uploaded image and the query '{user_query}', provide fitness-related advice or suggestions. "
              "The image shows a fitness-related activity or equipment. Assist the user with workout tips, form correction, "
              "or equipment usage guidance.")
    response = model.generate_content([sample_file, prompt])
    print("\nAssistant response for image query:")
    print(response.text)

# Main function to choose the query modality
def main():
    while True:
        print("\nChoose query mode:")
        print("1 - Text query")
        print("2 - Voice query")
        print("3 - Image query")
        print("q - Quit")
        choice = input("Enter your choice: ").strip().lower()

        if choice == "1":
            user_query = input("Enter your text query: ").strip()
            query(user_query)
        elif choice == "2":
            voice_query = get_voice_query()
            if voice_query:
                query(voice_query)
        elif choice == "3":
            handle_image_query()
        elif choice == "q":
            print("Exiting application.")
            sys.exit(0)
        else:
            print("Invalid choice. Please select 1, 2, 3 or q.")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:


import os
import re
import sys
import time
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# API and Model Imports
import google.generativeai as genai
from PIL import Image
import speech_recognition as sr
from googletrans import Translator

# LangChain Imports
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM, BaseLLM
from langchain.schema import Generation, LLMResult, SystemMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Configure Google API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = "AIzaSyCgaz3OFtXuNx-SCRPz2N58UCfpo0pcH_g"  # Fallback key
    print("Warning: Using fallback GOOGLE_API_KEY. Better to set it in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize translator
translator = Translator()

# Regular expression for parsing actions
ACTION_RE = re.compile(r'^Action:\s*(\w+)\s*:\s*(.+)$')

# System prompt for the medical chatbot
SYSTEM_PROMPT = """
You are an expert medical doctor with extensive experience in diagnosing and treating patient queries. Your role is to engage patients in a structured, empathetic, and conversational flow to identify the root cause of their health issues—such as illnesses, aches, or discomforts—and provide tailored solutions. These solutions should include appropriate remedies, medication suggestions (with general disclaimers if needed), lifestyle advice, diet plans, and exercise recommendations to promote overall health and recovery.

For example:

Patient: "I've been having a fever."
Doctor: "How long have you been experiencing this fever?"
Patient: "For three days."
Doctor: "Are you also experiencing symptoms like a dry cough, runny nose, or body aches?"
Patient: "No, doctor."
Doctor: "Alright, let's narrow this down. Have you noticed any other symptoms, like chills, fatigue, or loss of appetite? Also, have you been able to measure your temperature?"
Your responses should:

Ask follow-up questions to gather more details about symptoms, duration, severity, and any relevant medical history.
Analyze the information provided to suggest possible causes (e.g., viral infection, bacterial issue, etc.).
Recommend remedies (e.g., rest, hydration), over-the-counter medications if appropriate (e.g., paracetamol for fever), and caution about consulting a healthcare provider for prescriptions or severe cases.
Provide a simple, practical diet plan to support recovery (e.g., light meals, hydration tips) and address irregular eating habits for long-term health.
Suggest basic exercises or wellness tips (e.g., light stretching or breathing exercises) suitable for their condition.
Maintain a professional yet caring tone, ensuring the patient feels heard and supported.
If information is missing, politely prompt the patient to clarify. Avoid making definitive diagnoses requiring in-person tests, and include disclaimers like 'Please consult a local doctor if symptoms persist or worsen.' Ensure all advice aligns with general medical knowledge and promotes a healthy lifestyle.
""".strip()

# Image analysis prompt
IMAGE_ANALYSIS_PROMPT = """
Analyze this medical image and provide a professional assessment based on what you can observe. 
Consider potential conditions related to:
1. Skin conditions (rashes, wounds, infections)
2. Visible injuries or abnormalities
3. Any notable medical features

In your response:
- Describe what you observe in the image
- Suggest possible conditions or diagnoses based on the visual information
- Recommend appropriate next steps or home care if applicable
- Emphasize when in-person medical consultation is necessary
- Include appropriate medical disclaimers

Note: This analysis is not a definitive medical diagnosis and should be followed up with a healthcare professional.
"""

class GoogleGenAI(LLM):
    """Custom LangChain LLM class for Google's Generative AI models"""
    
    model_name: str = "gemini-1.5-flash"
    
    def __init__(self, model_name=None):
        super().__init__()
        if model_name:
            self.model_name = model_name
    
    @property
    def _llm_type(self) -> str:
        return "google-generativeai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the Google Generative AI model with the given prompt."""
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Async version of _call method."""
        return self._call(prompt, stop)


class Chatbot:
    """Medical chatbot with conversation history management."""
    
    def __init__(self, system_prompt: str, model_name: str = "gemini-1.5-flash"):
        """Initialize the chatbot with a system prompt and model."""
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
    
    def __call__(self, message: str) -> str:
        """Process a user message and get a response."""
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result
    
    def execute(self) -> str:
        """Generate a response based on the conversation history."""
        # Prepare prompt from the conversation history
        prompt = "\n".join([f'{msg["role"]}:{msg["content"]}' for msg in self.messages])
        model = genai.GenerativeModel(self.model_name)
        raw_response = model.generate_content(prompt)
        # Extract text content from the first candidate
        result_text = raw_response.candidates[0].content.parts[0].text
        return result_text
    
    def clear_history(self):
        """Clear conversation history but keep the system prompt."""
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})


class MedicalAssistant:
    """Main class for the medical assistant application with text, voice, and image capabilities."""
    
    def __init__(self):
        """Initialize the medical assistant."""
        self.chatbot = Chatbot(SYSTEM_PROMPT)
        self.action_functions = {
            "generate_workout": self.generate_workout,
            "suggest_meal": self.suggest_meal,
        }
    
    def generate_workout(self, level: str) -> str:
        """Generate a workout plan based on fitness level."""
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Generate a detailed workout plan for a {level} fitness level that is medically safe.
        Include:
        - Warm-up exercises
        - Main workout routine with repetitions and sets
        - Cool-down stretches
        - Safety precautions
        - Duration and frequency recommendations
        """
        response = model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text
    
    def suggest_meal(self, preferences: str) -> str:
        """Suggest a meal plan based on user preferences."""
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Create a nutritionally balanced meal plan with {preferences}.
        Include:
        - Breakfast, lunch, dinner, and snacks
        - Nutritional benefits of each suggestion
        - Portion size guidelines
        - Preparation tips
        - Alternatives for common dietary restrictions
        """
        response = model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text
    
    def analyze_medical_image(self, image_path: str, user_query: str = "") -> str:
        """Analyze a medical image with optional user query."""
        try:
            # Validate image file
            if not os.path.isfile(image_path):
                return "Error: File not found. Please check the path and try again."
            
            # Upload the image to Google's API
            sample_file = genai.upload_file(path=image_path, display_name="Medical Image")
            
            # Create a prompt that includes both the analysis prompt and user query
            prompt = IMAGE_ANALYSIS_PROMPT
            if user_query:
                prompt += f"\n\nUser question about the image: {user_query}"
            
            # Use the Pro model for image analysis
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
            response = model.generate_content([sample_file, prompt])
            
            return response.text
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def get_voice_input(self) -> Optional[str]:
        """Capture voice input from microphone and convert to text."""
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        print("\nPlease speak your health concern or question...")
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=10)
            
            voice_text = recognizer.recognize_google(audio)
            print(f"Recognized: \"{voice_text}\"")
            return voice_text
        except sr.UnknownValueError:
            print("Sorry, could not understand your speech.")
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
        except Exception as e:
            print(f"Error capturing voice: {e}")
        return None
    
    def process_text_query(self, query: str, max_turns: int = 5) -> None:
        """Process a text query with multi-turn conversation support."""
        i = 0
        next_prompt = query
        
        print("\n--- Starting medical consultation ---")
        while i < max_turns:
            i += 1
            
            # Get response from chatbot
            result = self.chatbot(next_prompt)
            print("\nMedical Assistant:")
            print(result)
            
            # Look for action commands in the response
            actions = [ACTION_RE.match(a) for a in result.split('\n') if ACTION_RE.match(a)]
            if actions:
                action, action_input = actions[0].groups()
                if action not in self.action_functions:
                    print(f"\nWarning: Unknown action requested: {action}")
                    break
                
                print(f"\n--- Performing: {action} ({action_input.strip()}) ---")
                observation = self.action_functions[action](action_input.strip())
                print(f"\nResults:\n{observation}\n")
                
                next_prompt = f"I've received this information: {observation}\nPlease continue with your medical advice."
                time.sleep(1)  # Brief pause before next turn
            else:
                # Ask for user's next input or end conversation
                user_input = input("\nYour response (or type 'exit' to end): ").strip()
                if user_input.lower() in ['exit', 'quit', 'end']:
                    break
                next_prompt = user_input
        
        print("\n--- End of consultation ---")
        # Clear conversation history for next session
        self.chatbot.clear_history()
    
    def show_welcome_message(self):
        """Display a welcome message with instructions."""
        print("\n" + "="*60)
        print("          MEDICAL ASSISTANT - Health Consultation Bot")
        print("="*60)
        print("This application allows you to consult about health concerns through:")
        print("  - Text conversation with a medical AI")
        print("  - Voice input for your queries")
        print("  - Medical image analysis (skin conditions, wounds, etc.)")
        print("\nIMPORTANT: This is not a substitute for professional medical advice.")
        print("           Always consult a healthcare provider for serious concerns.")
        print("="*60 + "\n")
    
    def run(self):
        """Run the main application loop."""
        self.show_welcome_message()
        
        while True:
            print("\nChoose consultation mode:")
            print("1 - Text consultation")
            print("2 - Voice consultation")
            print("3 - Image analysis")
            print("q - Quit application")
            
            choice = input("\nSelect option: ").strip().lower()
            
            if choice == "1":
                user_query = input("\nDescribe your health concern: ").strip()
                self.process_text_query(user_query)
            
            elif choice == "2":
                voice_query = self.get_voice_input()
                if voice_query:
                    self.process_text_query(voice_query)
            
            elif choice == "3":
                self.handle_image_consultation()
            
            elif choice == "q":
                print("\nThank you for using Medical Assistant. Stay healthy!")
                break
            
            else:
                print("Invalid option. Please select 1, 2, 3, or q.")
    
    def handle_image_consultation(self):
        """Handle the image-based medical consultation."""
        image_path = input("\nEnter the path to the medical image file: ").strip()
        
        # Validate path
        if not os.path.exists(image_path):
            print("Error: File not found. Please check the path and try again.")
            return
        
        # Show the image if possible
        try:
            image = Image.open(image_path)
            image.show()  # Opens the image in default viewer
        except Exception as e:
            print(f"Note: Unable to display image locally ({e}), but will still analyze it.")
        
        # Get additional context from user
        user_query = input("\nDescribe your concern or ask a question about this image: ").strip()
        
        # Analyze the image
        print("\nAnalyzing medical image... Please wait.")
        analysis = self.analyze_medical_image(image_path, user_query)
        
        print("\nMedical Image Analysis:")
        print("-" * 40)
        print(analysis)
        print("-" * 40)
        print("\nReminder: This analysis is not a definitive diagnosis.")
        print("Please consult a healthcare professional for proper medical advice.")


if __name__ == "__main__":
    # Create and run the medical assistant
    assistant = MedicalAssistant()
    assistant.run()

