
import streamlit as st

st.title("Translation Chat App")

import os
import sys
import re
import requests
import speech_recognition as sr
from typing import List, Optional, Union
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.agents import AgentExecutor, AgentOutputParser, LLMSingleActionAgent, Tool
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage
from langchain.llms.base import LLM
from langchain.prompts import StringPromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from PIL import Image
from langchain.schema import AgentAction, AgentFinish

# Set API keys directly
os.environ["TAVILY_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCgaz3OFtXuNx-SCRPz2N58UCfpo0pcH_g"
os.environ["SERPER_API_KEY"] = "ed4acec1529a6f8755a04900d2554b5252aba850b59262e44712c7596509ef4a"
os.environ["AZURE_MAPS_KEY"] = "EumXcWSYqKLcsw9zymB1cPRIfDzNbZBXO7BCjKsbsAITXSpRIZbMJQQJ99BBACYeBjFPDDZUAAAgAZMP1DsH"

# Load environment variables
load_dotenv()

# Verify API keys are set
required_keys = ["TAVILY_API_KEY", "GOOGLE_API_KEY", "SERPER_API_KEY", "AZURE_MAPS_KEY"]
for key in required_keys:
    if not os.getenv(key):
        sys.exit(f"Error: {key} not found in environment variables. Please set it in .env file.")

# Configure Google API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

import os
import sys
import re
import requests
import speech_recognition as sr
from typing import List, Optional, Union
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.agents import AgentExecutor, AgentOutputParser, LLMSingleActionAgent, Tool
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage
from langchain.llms.base import LLM
from langchain.prompts import StringPromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from PIL import Image
from langchain.schema import AgentAction, AgentFinish

# Verify API keys are set
required_keys = ["TAVILY_API_KEY", "GOOGLE_API_KEY", "SERPER_API_KEY", "AZURE_MAPS_KEY"]
for key in required_keys:
    if not os.getenv(key):
        sys.exit(f"Error: {key} not found in environment variables. Please set it in .env file.")

# Configure Google API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize DuckDuckGo search tool
search_tool = DuckDuckGoSearchRun()

# System Prompts
SYSTEM_PROMPT = """
You are an expert medical doctor providing empathetic and structured health advice. Engage patients conversationally based on the chat history to identify their health issues (e.g., illness, aches) and offer tailored solutions, including remedies, medication suggestions (with disclaimers), diet plans, and exercises.

Example:
Patient: "I've been having a fever."
Assistant: "How long have you had this fever?"
Patient: "Three days."
Assistant: "Are you experiencing symptoms like cough or fatigue?"

When appropriate, use actions in this format:
- "Action: find_doctors : condition near location"
- "Action: find_pharmacies : location"
- "Action: analyze_medical_image : image_path"

If the patient provides a location and condition, suggest doctors immediately using the `find_doctors` action. Include disclaimers: 'Consult a healthcare provider for serious issues. I cannot diagnose your condition; my advice is for guidance only.'
Chat history is provided below for context.
"""

DOCTOR_AGENT_PROMPT = """
You are a medical referral specialist. Analyze the patient's condition and location from the input and chat history to recommend appropriate specialists. Use search tools to find real doctors, providing specialty type, rationale, and contact details if available.
"""

PHARMACY_AGENT_PROMPT = """
You are a pharmacy specialist. Recommend over-the-counter medications, pharmacy services, and use search tools to find real pharmacies near the patient's location based on the input and chat history. Include disclaimers about prescriptions.
"""

IMAGE_ANALYSIS_PROMPT = """
Analyze this medical image professionally, suggesting possible conditions (e.g., skin issues like rashes or scars, injuries). Describe observations, recommend next steps, and emphasize consulting a doctor. Note: This is not a definitive diagnosis. Only process images related to medical conditions.
"""

# Azure Maps API Functions
def get_location_coordinates(location):
    url = "https://atlas.microsoft.com/search/address/json"
    params = {
        "api-version": "1.0",
        "subscription-key": os.getenv("AZURE_MAPS_KEY"),
        "query": location,
        "limit": 1
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("results") and len(data["results"]) > 0:
            position = data["results"][0]["position"]
            return position["lat"], position["lon"]
        return None, None
    except requests.RequestException as e:
        print(f"Azure Maps geocoding error: {str(e)}")
        return None, None

def azure_maps_search_poi(lat, lon, search_term, entity_type, radius=10000, limit=5):
    search_url = "https://atlas.microsoft.com/search/poi/json"
    if entity_type == "doctors":
        search_term = f"{search_term} doctor medical"
    elif entity_type == "pharmacies":
        search_term = f"pharmacy {search_term}"
    search_params = {
        "api-version": "1.0",
        "subscription-key": os.getenv("AZURE_MAPS_KEY"),
        "query": search_term,
        "lat": lat,
        "lon": lon,
        "radius": radius,
        "limit": limit
    }
    try:
        search_response = requests.get(search_url, params=search_params)
        search_response.raise_for_status()
        search_data = search_response.json()
        if not search_data.get("results") or len(search_data["results"]) == 0:
            return f"No {entity_type} found for '{search_term}' near the specified location."
        results = []
        for i, result in enumerate(search_data["results"], 1):
            poi = result.get("poi", {})
            address = result.get("address", {})
            phone = poi.get("phone", "No phone number available")
            street = f"{address.get('streetNumber', '')} {address.get('streetName', '')}".strip()
            locality = address.get('localName', '') or address.get('municipality', '')
            region = address.get('countrySubdivision', '')
            full_address = ", ".join(filter(None, [street, locality, region])) or "Address not available"
            result_str = f"{i}. {poi.get('name', 'Unnamed')}\n   Address: {full_address}\n   Phone: {phone}\n"
            categories = poi.get('categories', [])
            if categories:
                result_str += f"   Category: {', '.join(categories)}\n"
            results.append(result_str)
        return "\n".join(results)
    except requests.RequestException as e:
        return f"Azure Maps search error: {str(e)}"

def azure_maps_search(query, entity_type, limit=5):
    parts = query.split("near")
    if len(parts) != 2:
        return f"Please specify a search in the format 'condition near location'."
    search_term = parts[0].strip()
    location = parts[1].strip()
    lat, lon = get_location_coordinates(location)
    if not lat or not lon:
        return f"Location '{location}' not found. Please try a different location."
    return azure_maps_search_poi(lat, lon, search_term, entity_type, limit=limit)

# Search Functions
def tavily_search(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    url = "https://api.tavily.com/search"
    params = {"api_key": api_key, "query": query, "max_results": 5}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            return "No results found with Tavily."
        return "\n".join([f"{i}. {r.get('title', 'No title')}: {r.get('content', 'No content')[:200]}...\n   {r.get('url', 'No URL')}" for i, r in enumerate(results, 1)])
    except requests.RequestException as e:
        return f"Tavily search error: {str(e)}"

def serper_search(query: str) -> str:
    api_key = os.getenv("SERPER_API_KEY")
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": 5}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("organic", [])
        if not results:
            return "No results found with Serper."
        return "\n".join([f"{i}. {r.get('title', 'No title')}: {r.get('snippet', 'No snippet')}\n   {r.get('link', 'No link')}" for i, r in enumerate(results, 1)])
    except requests.RequestException as e:
        return f"Serper search error: {str(e)}"

def multi_search(query: str) -> str:
    results = []
    for func in [search_tool.run, tavily_search, serper_search]:
        try:
            result = func(query)
            results.append(result)
        except Exception as e:
            results.append(f"Search failed: {str(e)}")
    combined = "\n\n".join([f"--- Source {i+1} ---\n{r}" for i, r in enumerate(results) if r])
    if len(combined) > 2000:
        model = genai.GenerativeModel("gemini-1.5-flash")
        try:
            summary = model.generate_content(f"Summarize this search result:\n{combined[:10000]}").text
            return f"Summary:\n{summary}"
        except Exception as e:
            return f"Summary failed: {str(e)}\nTruncated Results:\n{combined[:2000]}"
    return combined

# Custom LLM for LangChain
class GoogleGenAI(LLM):
    model_name: str = "gemini-1.5-flash"
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        model = genai.GenerativeModel(self.model_name)
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"LLM error: {str(e)}"
    @property
    def _llm_type(self) -> str:
        return "google-generativeai"

# Agent Utilities
class CustomAgentOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            final_answer = llm_output.split("Final Answer:")[-1].strip()
            return AgentFinish(return_values={"output": final_answer}, log=llm_output)
        regex = r"Action:\s*(\w+)\s*:\s*(.+)"
        match = re.search(regex, llm_output, re.DOTALL)
        if match:
            action, input_str = match.group(1).strip(), match.group(2).strip()
            return AgentAction(tool=action, tool_input=input_str, log=llm_output)
        return AgentFinish(return_values={"output": llm_output.strip()}, log=llm_output)

class CustomPromptTemplate(StringPromptTemplate):
    template: str = """
    {agent_prompt}
    Tools:
    {tools}
    Chat History:
    {chat_history}
    Question: {input}
    Agent Scratchpad:
    {agent_scratchpad}
    """
    tools: List[Tool]
    agent_prompt: str
    def format(self, **kwargs) -> str:
        kwargs["agent_prompt"] = self.agent_prompt
        kwargs["tools"] = "\n".join([f"{t.name}: {t.description}" for t in self.tools])
        kwargs["chat_history"] = kwargs.get("chat_history", "")
        kwargs["agent_scratchpad"] = kwargs.get("agent_scratchpad", "")
        return self.template.format(**kwargs)

# Agent Setup
def get_doctor_agent(llm: LLM, memory: ConversationBufferMemory):
    tools = [
        Tool(name="search_doctors", func=lambda q: multi_search(f"doctors {q}"), description="Search for doctors based on a general query."),
        Tool(name="find_doctors_nearby", func=lambda q: azure_maps_search(q, "doctors"), description="Find doctors for a condition near a location using Azure Maps. Input format: 'condition near location'.")
    ]
    prompt = CustomPromptTemplate(tools=tools, agent_prompt=DOCTOR_AGENT_PROMPT, input_variables=["input", "chat_history", "agent_scratchpad"])
    agent = LLMSingleActionAgent(llm_chain=LLMChain(llm=llm, prompt=prompt), output_parser=CustomAgentOutputParser(), stop=["\nObservation:"], allowed_tools=[t.name for t in tools])
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False)

def get_pharmacy_agent(llm: LLM, memory: ConversationBufferMemory):
    tools = [
        Tool(name="search_pharmacies", func=lambda q: multi_search(f"pharmacies {q}"), description="Search for pharmacies based on a general query."),
        Tool(name="find_pharmacies_nearby", func=lambda q: azure_maps_search(q, "pharmacies"), description="Find pharmacies near a location using Azure Maps. Input format: 'service near location'.")
    ]
    prompt = CustomPromptTemplate(tools=tools, agent_prompt=PHARMACY_AGENT_PROMPT, input_variables=["input", "chat_history", "agent_scratchpad"])
    agent = LLMSingleActionAgent(llm_chain=LLMChain(llm=llm, prompt=prompt), output_parser=CustomAgentOutputParser(), stop=["\nObservation:"], allowed_tools=[t.name for t in tools])
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False)
    
# Medical Assistant Class
class MedicalAssistant:
    def __init__(self):
        self.llm = GoogleGenAI()
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
        self.doctor_agent = get_doctor_agent(self.llm, self.memory)
        self.pharmacy_agent = get_pharmacy_agent(self.llm, self.memory)
        self.actions = {
            "find_doctors": self.find_doctors,
            "find_pharmacies": self.find_pharmacies,
            "analyze_medical_image": self.analyze_medical_image
        }
        self.user_lang = None
        self.interface_texts = {}

    def translate(self, text, source_lang, target_lang):
        """Translate text using the LLM."""
        if source_lang == target_lang:
            return text
        prompt = f"Translate the following medical text from {source_lang} to {target_lang}: {text}"
        translated_text = self.llm._call(prompt)
        if translated_text.startswith("LLM error"):
            print(translated_text)
            return text
        return translated_text

    def find_doctors(self, query: str) -> str:
        try:
            azure_results = azure_maps_search(query, "doctors")
            if "error" not in azure_results.lower() and "not found" not in azure_results.lower() and "please specify" not in azure_results.lower():
                return f"Found doctors using Azure Maps:\n\n{azure_results}\n\nDisclaimer: Consult a healthcare provider for professional medical advice."
            result = self.doctor_agent.invoke({"input": query, "chat_history": self.memory.buffer})["output"]
            return f"{result}\n\nDisclaimer: Consult a healthcare provider for professional medical advice."
        except Exception as e:
            print(f"Doctor search error: {str(e)}")
            web_search_result = multi_search(f"doctors for {query}")
            return f"Could not find doctors through maps due to an error. Here are web search results:\n\n{web_search_result}\n\nDisclaimer: Consult a healthcare provider for professional medical advice."

    def find_pharmacies(self, query: str) -> str:
        try:
            if "near" not in query.lower():
                query = f"pharmacies near {query}"
            azure_results = azure_maps_search(query, "pharmacies")
            if "error" not in azure_results.lower() and "not found" not in azure_results.lower() and "please specify" not in azure_results.lower():
                return f"Found pharmacies using Azure Maps:\n\n{azure_results}\n\nDisclaimer: Consult a pharmacist or doctor before taking medications."
            result = self.pharmacy_agent.invoke({"input": query, "chat_history": self.memory.buffer})["output"]
            return f"{result}\n\nDisclaimer: Consult a pharmacist or doctor before taking medications."
        except Exception as e:
            print(f"Pharmacy search error: {str(e)}")
            web_search_result = multi_search(f"pharmacies in {query}")
            return f"Could not find pharmacies through maps due to an error. Here are web search results:\n\n{web_search_result}\n\nDisclaimer: Consult a pharmacist or doctor before taking medications."

    def analyze_medical_image(self, image_path: str) -> str:
        if not os.path.isfile(image_path):
            error_msg = "Error: Image file not found."
            return self.translate(error_msg, "en", self.user_lang)
        try:
            print("Uploading and analyzing medical image...")
            file = genai.upload_file(path=image_path, display_name="Medical Image")
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            last_user_input = next((m.content for m in reversed(self.memory.chat_memory.messages) if isinstance(m, HumanMessage)), "")
            prompt = IMAGE_ANALYSIS_PROMPT + (f"\nUser query: {last_user_input}" if last_user_input else "")
            response_en = model.generate_content([file, prompt]).text
            response = self.translate(response_en, "en", self.user_lang)
            disclaimer_en = "Disclaimer: This is not a definitive diagnosis. Consult a doctor."
            disclaimer = self.translate(disclaimer_en, "en", self.user_lang)
            return f"{response}\n\n{disclaimer}"
        except Exception as e:
            error_msg = f"Image analysis error: {str(e)}"
            return self.translate(error_msg, "en", self.user_lang)

    def get_voice_input(self) -> Optional[str]:
        recognizer = sr.Recognizer()
        language_codes = {
            "en": "en-US",
            "es": "es-ES",
            "fr": "fr-FR",
            "de": "de-DE",
            # Add more as needed
        }
        language = language_codes.get(self.user_lang, "en-US")
        try:
            with sr.Microphone() as source:
                print(self.translate("Speak your health concern (10 seconds)...", "en", self.user_lang))
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            return recognizer.recognize_google(audio, language=language)
        except sr.RequestError as e:
            error_msg = f"Voice recognition service error: {str(e)}"
            return self.translate(error_msg, "en", self.user_lang)
        except sr.UnknownValueError:
            error_msg = "Could not understand audio."
            return self.translate(error_msg, "en", self.user_lang)
        except Exception as e:
            error_msg = f"Voice input error: {str(e)}"
            return self.translate(error_msg, "en", self.user_lang)

    def extract_condition_and_location(self, query: str) -> tuple:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Extract the health condition and location from this query.
        If no location is mentioned, return 'None' for location.
        If no condition is mentioned, return 'None' for condition.
        Respond with:
        condition: [condition]
        location: [location]
        Query: "{query}"
        """
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            condition_match = re.search(r'condition:\s*(.+)', text)
            location_match = re.search(r'location:\s*(.+)', text)
            condition = condition_match.group(1).strip() if condition_match else "None"
            location = location_match.group(1).strip() if location_match else "None"
            return condition, location
        except Exception as e:
            print(f"Extraction error: {str(e)}")
            return "None", "None"

    def process_query(self, query: str):
        query_en = self.translate(query, self.user_lang, "en")
        if query_en.lower() == "clear history":
            self.memory.clear()
            print(self.translate("Conversation history cleared.", "en", self.user_lang))
            return
        self.memory.chat_memory.add_message(HumanMessage(content=query_en))
        condition, location = self.extract_condition_and_location(query_en)
        if condition.lower() != "none" and location.lower() != "none":
            doctor_query = f"{condition} near {location}"
            result_en = self.find_doctors(doctor_query)
            result = self.translate(result_en, "en", self.user_lang)
            print(f"\n{self.translate('Suggested Doctors:', 'en', self.user_lang)}\n{result}")
            self.memory.chat_memory.add_message(AIMessage(content=result_en))
        else:
            prompt = f"{SYSTEM_PROMPT}\nChat History:\n{self.memory.buffer}\n\nUser Query: {query_en}"
            response_en = self.llm._call(prompt)
            response = self.translate(response_en, "en", self.user_lang)
            print(f"\n{self.translate('Assistant:', 'en', self.user_lang)} {response}")
            self.memory.chat_memory.add_message(AIMessage(content=response_en))
            for line in response_en.split('\n'):
                match = re.match(r"Action:\s*(\w+)\s*:\s*(.+)", line.strip())
                if match:
                    action, input_str = match.groups()
                    if action in self.actions:
                        result_en = self.actions[action](input_str.strip())
                        result = self.translate(result_en, "en", self.user_lang)
                        print(f"\n{self.translate('Action Result:', 'en', self.user_lang)} {result}")
                        self.memory.chat_memory.add_message(AIMessage(content=f"Action {action} result: {result_en}"))

    def run(self):
        print("=== Medical Assistant ===")
        print("Please specify your preferred language (e.g., English, Spanish, French).")
        lang_input = input("Language: ").strip()
        prompt = f"What is the ISO 639-1 code for the language '{lang_input}'?"
        try:
            response = self.llm._call(prompt)
            self.user_lang = response.strip().lower()
            print(f"Language set to {self.user_lang}")
        except Exception as e:
            print(f"Error setting language: {str(e)}. Defaulting to English.")
            self.user_lang = "en"

        # Translate interface texts
        interface_texts_en = {
            "options_menu": "Options: 1-Text, 2-Voice, 3-Image, q-Quit",
            "select": "Select: ",
            "health_concern": "Your health concern: ",
            "image_path": "Image path: ",
            "goodbye": "Goodbye.",
            "invalid_option": "Invalid option. Choose 1, 2, 3, or q.",
            "clear_history_note": "Type 'clear history' during text input to reset the conversation."
        }
        self.interface_texts = {key: self.translate(text, "en", self.user_lang) for key, text in interface_texts_en.items()}

        print(self.interface_texts["options_menu"])
        print(self.interface_texts["clear_history_note"])

        while True:
            choice = input(self.interface_texts["select"]).strip().lower()
            if choice == "1":
                query = input(self.interface_texts["health_concern"]).strip()
                if query:
                    self.process_query(query)
                else:
                    print(self.translate("Please enter a valid query.", "en", self.user_lang))
            elif choice == "2":
                print(self.translate("Initiating voice input...", "en", self.user_lang))
                query = self.get_voice_input()
                if query and "error" not in query.lower():
                    print(f"{self.translate('Recognized:', 'en', self.user_lang)} {query}")
                    self.process_query(query)
                else:
                    print(query or self.translate("Voice input failed.", "en", self.user_lang))
            elif choice == "3":
                path = input(self.interface_texts["image_path"]).strip()
                if path:
                    print(self.translate("Analyzing medical image...", "en", self.user_lang))
                    result = self.analyze_medical_image(path)
                    print(f"\n{self.translate('Analysis:', 'en', self.user_lang)} {result}")
                    self.memory.chat_memory.add_message(AIMessage(content=result))
                else:
                    print(self.translate("Please provide a valid image path.", "en", self.user_lang))
            elif choice == "q":
                print(self.interface_texts["goodbye"])
                break
            else:
                print(self.interface_texts["invalid_option"])

if __name__ == "__main__":
    assistant = MedicalAssistant()
    assistant.run()





# User input
user_input = st.text_area("Enter text to translate:")

if st.button("Translate"):
    if user_input:
        try:
            translated_text = translate_text(user_input)  # Assuming a function named translate_text exists
            st.success(f"Translated Text: {translated_text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter some text to translate.")
