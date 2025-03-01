
import os
import sys
import re
import requests
import speech_recognition as sr
from typing import List, Optional, Union
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.agents import AgentExecutor, AgentOutputParser, LLMSingleActionAgent,Tool
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage
from langchain.llms.base import LLM
from langchain.prompts import StringPromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from PIL  import Image 
import speech_recognition as sr 
from langchain.schema import AgentAction, AgentFinish, HumanMessage, AIMessage


os.environ["TAVILY_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCgaz3OFtXuNx-SCRPz2N58UCfpo0pcH_g"
os.environ["SERPER_API_KEY"] = "ed4acec1529a6f8755a04900d2554b5252aba850b59262e44712c7596509ef4a"
os.environ["AZURE_MAPS_KEY"] = "EumXcWSYqKLcsw9zymB1cPRIfDzNbZBXO7BCjKsbsAITXSpRIZbMJQQJ99BBACYeBjFPDDZUAAAgAZMP1DsH"


load_dotenv()

AZURE_MAPS_KEY = "EumXcWSYqKLcsw9zymB1cPRIfDzNbZBXO7BCjKsbsAITXSpRIZbMJQQJ99BBACYeBjFPDDZUAAAgAZMP1DsH"
os.environ["AZURE_MAPS_KEY"] = AZURE_MAPS_KEY
required_keys = ["TAVILY_API_KEY", "GOOGLE_API_KEY", "SERPER_API_KEY"]
for key in required_keys:
    if not os.getenv(key):
        sys.exit(f"Error: {key} not found in environment variables. Please set it in .env file.")




genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

search_tool = DuckDuckGoSearchRun()

SYSTEM_PROMPT = """
You are an expert medical doctor providing empathetic and structured health advice. Engage patients conversationally based on the chat history to identify their health issues (e.g., illness, aches) and offer tailored solutions, including remedies, medication suggestions (with disclaimers), diet plans, and exercises.

Example:
Patient: "I've been having a fever."
Assistant: "How long have you had this fever?"
Patient: "Three days."
Assistant: "Are you experiencing symptoms like cough or fatigue?"

When appropriate, use actions in this format:
- "Action: find_doctors : condition near location" (e.g., "Action: find_doctors : fever near OMR")
- "Action: find_pharmacies : location" (e.g., "Action: find_pharmacies : OMR")
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
Analyze this medical image professionally, suggesting possible conditions (e.g., skin issues, injuries). Describe observations, recommend next steps, and emphasize consulting a doctor. Note: This is not a definitive diagnosis.
"""


def get_location_coordinates(location):
    """Get coordinates for a location using Azure Maps."""
    url = "https://atlas.microsoft.com/search/address/json"
    
    params = {
        "api-version": "1.0",
        "subscription-key": AZURE_MAPS_KEY,
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
        else:
            print(f"No coordinates found for location: {location}")
            return None, None
    except requests.RequestException as e:
        print(f"Azure Maps geocoding error: {str(e)}")
        return None, None

def azure_maps_search_poi(lat, lon, search_term, entity_type, radius=10000, limit=5):
    """Search for points of interest using Azure Maps."""
    search_url = "https://atlas.microsoft.com/search/poi/json"
    
    category_param = ""
    if entity_type == "doctors":
        search_term = f"{search_term} doctor medical"
    elif entity_type == "pharmacies":
        search_term = f"pharmacy {search_term}"
    
    search_params = {
        "api-version": "1.0",
        "subscription-key": AZURE_MAPS_KEY,
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
            
            # Format phone number if available
            phone = poi.get("phone", "No phone number available")
            
            # Construct address
            street = f"{address.get('streetNumber', '')} {address.get('streetName', '')}".strip()
            locality = address.get('localName', '') or address.get('municipality', '')
            region = address.get('countrySubdivision', '')
            
            full_address = ", ".join(filter(None, [street, locality, region]))
            if not full_address:
                full_address = "Address not available"
            
            # Format result
            result_str = f"{i}. {poi.get('name', 'Unnamed')}\n"
            result_str += f"   Address: {full_address}\n"
            result_str += f"   Phone: {phone}\n"
            
            # Add categories if available
            categories = poi.get('categories', [])
            if categories:
                result_str += f"   Category: {', '.join(categories)}\n"
            
            results.append(result_str)
        
        return "\n".join(results)
    except requests.RequestException as e:
        return f"Azure Maps search error: {str(e)}"

def azure_maps_search(query, entity_type, limit=5):
    """Parse query and search for doctors or pharmacies."""
    # Parse query to extract condition and location
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
    """Search using Tavily API."""
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
    """Search using Serper API."""
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
    """Combine results from multiple search engines."""
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


# In[ ]:


# Azure Maps API Functions
def get_location_coordinates(location):
    """Get coordinates for a location using Azure Maps."""
    url = "https://atlas.microsoft.com/search/address/json"
    
    params = {
        "api-version": "1.0",
        "subscription-key": AZURE_MAPS_KEY,
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
        else:
            print(f"No coordinates found for location: {location}")
            return None, None
    except requests.RequestException as e:
        print(f"Azure Maps geocoding error: {str(e)}")
        return None, None

def azure_maps_search_poi(lat, lon, search_term, entity_type, radius=10000, limit=5):
    """Search for points of interest using Azure Maps."""
    search_url = "https://atlas.microsoft.com/search/poi/json"

    category_param = ""
    if entity_type == "doctors":
        # Use search term to find medical facilities
        search_term = f"{search_term} doctor medical"
    elif entity_type == "pharmacies":
        search_term = f"pharmacy {search_term}"
    
    search_params = {
        "api-version": "1.0",
        "subscription-key": AZURE_MAPS_KEY,
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
            
            # Format phone number if available
            phone = poi.get("phone", "No phone number available")
            
            # Construct address
            street = f"{address.get('streetNumber', '')} {address.get('streetName', '')}".strip()
            locality = address.get('localName', '') or address.get('municipality', '')
            region = address.get('countrySubdivision', '')
            
            full_address = ", ".join(filter(None, [street, locality, region]))
            if not full_address:
                full_address = "Address not available"
            
            # Format result
            result_str = f"{i}. {poi.get('name', 'Unnamed')}\n"
            result_str += f"   Address: {full_address}\n"
            result_str += f"   Phone: {phone}\n"
            
            # Add categories if available
            categories = poi.get('categories', [])
            if categories:
                result_str += f"   Category: {', '.join(categories)}\n"
            
            results.append(result_str)
        
        return "\n".join(results)
    except requests.RequestException as e:
        return f"Azure Maps search error: {str(e)}"

def azure_maps_search(query, entity_type, limit=5):
    """Parse query and search for doctors or pharmacies."""
    # Parse query to extract condition and location
    parts = query.split("near")
    
    if len(parts) != 2:
        return f"Please specify a search in the format 'condition near location'."
        
    search_term = parts[0].strip()
    location = parts[1].strip()
    
    # First get the coordinates for the location
    lat, lon = get_location_coordinates(location)
    
    if not lat or not lon:
        return f"Location '{location}' not found. Please try a different location."
    
    # Now search for POIs near the coordinates
    return azure_maps_search_poi(lat, lon, search_term, entity_type, limit=limit)

# Search Functions
def tavily_search(query: str) -> str:
    """Search using Tavily API."""
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
    """Search using Serper API."""
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
    """Combine results from multiple search engines."""
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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


# Agent Setup
def get_doctor_agent(llm: LLM, memory: ConversationBufferMemory):
    tools = [
        Tool(
            name="search_doctors",
            func=lambda q: multi_search(f"doctors {q}"),
            description="Search for doctors based on a general query."
        ),
        Tool(
            name="find_doctors_nearby",
            func=lambda q: azure_maps_search(q, "doctors"),
            description="Find doctors for a condition near a location using Azure Maps. Input format: 'condition near location'."
        )
    ]
    prompt = CustomPromptTemplate(
        tools=tools,
        agent_prompt=DOCTOR_AGENT_PROMPT,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )
    agent = LLMSingleActionAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        output_parser=CustomAgentOutputParser(),
        stop=["\nObservation:"],
        allowed_tools=[t.name for t in tools]
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False)

def get_pharmacy_agent(llm: LLM, memory: ConversationBufferMemory):
    tools = [
        Tool(
            name="search_pharmacies",
            func=lambda q: multi_search(f"pharmacies {q}"),
            description="Search for pharmacies based on a general query."
        ),
        Tool(
            name="find_pharmacies_nearby",
            func=lambda q: azure_maps_search(q, "pharmacies"),
            description="Find pharmacies near a location using Azure Maps. Input format: 'service near location'."
        )
    ]
    prompt = CustomPromptTemplate(
        tools=tools,
        agent_prompt=PHARMACY_AGENT_PROMPT,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )
    agent = LLMSingleActionAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        output_parser=CustomAgentOutputParser(),
        stop=["\nObservation:"],
        allowed_tools=[t.name for t in tools]
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False)


# In[ ]:


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

    def find_doctors(self, query: str) -> str:
        try:
            # First try Azure Maps
            azure_results = azure_maps_search(query, "doctors")
            if "error" not in azure_results.lower() and "not found" not in azure_results.lower() and "please specify" not in azure_results.lower():
                return f"Found doctors using Azure Maps:\n\n{azure_results}\n\nDisclaimer: Consult a healthcare provider for professional medical advice."
            
            # Fall back to agent if Azure Maps doesn't work
            result = self.doctor_agent.invoke({"input": query, "chat_history": self.memory.buffer})["output"]
            return f"{result}\n\nDisclaimer: Consult a healthcare provider for professional medical advice."
        except Exception as e:
            print(f"Doctor search error: {str(e)}")
            # Fall back to web search on any error
            web_search_result = multi_search(f"doctors for {query}")
            return f"Could not find doctors through maps due to an error. Here are web search results:\n\n{web_search_result}\n\nDisclaimer: Consult a healthcare provider for professional medical advice."

    def find_pharmacies(self, query: str) -> str:
        try:
            # Add "near" if not present
            if "near" not in query.lower():
                query = f"pharmacies near {query}"
                
            # First try Azure Maps
            azure_results = azure_maps_search(query, "pharmacies")
            if "error" not in azure_results.lower() and "not found" not in azure_results.lower() and "please specify" not in azure_results.lower():
                return f"Found pharmacies using Azure Maps:\n\n{azure_results}\n\nDisclaimer: Consult a pharmacist or doctor before taking medications."
            
            # Fall back to agent if Azure Maps doesn't work
            result = self.pharmacy_agent.invoke({"input": query, "chat_history": self.memory.buffer})["output"]
            return f"{result}\n\nDisclaimer: Consult a pharmacist or doctor before taking medications."
        except Exception as e:
            print(f"Pharmacy search error: {str(e)}")
            # Fall back to web search on any error
            web_search_result = multi_search(f"pharmacies in {query}")
            return f"Could not find pharmacies through maps due to an error. Here are web search results:\n\n{web_search_result}\n\nDisclaimer: Consult a pharmacist or doctor before taking medications."

    def analyze_medical_image(self, image_path: str) -> str:
        try:
            if not os.path.isfile(image_path):
                return "Error: Image file not found."
            file = genai.upload_file(path=image_path, display_name="Medical Image")
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            last_user_input = next((m.content for m in reversed(self.memory.chat_memory.messages) if isinstance(m, HumanMessage)), "")
            prompt = IMAGE_ANALYSIS_PROMPT + (f"\nUser query: {last_user_input}" if last_user_input else "")
            response = model.generate_content([file, prompt])
            return f"{response.text}\n\nDisclaimer: This is not a definitive diagnosis. Consult a doctor."
        except Exception as e:
            return f"Image analysis error: {str(e)}"

    def get_voice_input(self) -> Optional[str]:
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                print("Speak your health concern (10 seconds)...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            return recognizer.recognize_google(audio)
        except sr.RequestError as e:
            return f"Voice recognition service error: {str(e)}"
        except sr.UnknownValueError:
            return "Could not understand audio."
        except Exception as e:
            return f"Voice input error: {str(e)}"

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
        if query.lower() == "clear history":
            self.memory.clear()
            print("Conversation history cleared.")
            return

        # Save user input to memory
        self.memory.chat_memory.add_message(HumanMessage(content=query))

        # Extract condition and location
        condition, location = self.extract_condition_and_location(query)
        if condition.lower() != "none" and location.lower() != "none":
            doctor_query = f"{condition} near {location}"
            result = self.find_doctors(doctor_query)
            print(f"\nSuggested Doctors:\n{result}")
            self.memory.chat_memory.add_message(AIMessage(content=result))
        else:
            # Generate response using LLM
            prompt = f"{SYSTEM_PROMPT}\nChat History:\n{self.memory.buffer}\n\nUser Query: {query}"
            response = self.llm._call(prompt)
            print(f"\nAssistant: {response}")
            self.memory.chat_memory.add_message(AIMessage(content=response))

            # Check for actions in the response
            for line in response.split('\n'):
                match = re.match(r"Action:\s*(\w+)\s*:\s*(.+)", line.strip())
                if match:
                    action, input_str = match.groups()
                    if action in self.actions:
                        result = self.actions[action](input_str.strip())
                        print(f"\nAction Result: {result}")
                        self.memory.chat_memory.add_message(AIMessage(content=f"Action {action} result: {result}")) 

    def run(self):
        print("=== Medical Assistant ===")
        print("Options: 1-Text, 2-Voice, 3-Image, q-Quit")
        print("Type 'clear history' during text input to reset the conversation.")
        
        # Test Azure Maps connection at startup
        try:
            test_result = get_location_coordinates("New York")
            if not test_result[0]:
                print("\nWarning: Azure Maps geolocation test failed. Location-based searches may not work correctly.")
            else:
                print(f"\nAzure Maps connection successful. New York coordinates: {test_result}")
                
                # Test POI search
                test_poi = azure_maps_search_poi(test_result[0], test_result[1], "hospital", "doctors", limit=1)
                print(f"Test POI search result: {test_poi[:100]}...")
        except Exception as e:
            print(f"\nWarning: Azure Maps connection test failed: {str(e)}. Location-based searches may not work correctly.")
        
        while True:
            choice = input("\nSelect: ").strip().lower()
            if choice == "1":
                query = input("Your health concern: ").strip()
                if query:
                    self.process_query(query)
                else:
                    print("Please enter a valid query.")
            elif choice == "2":
                query = self.get_voice_input()
                if query and "error" not in query.lower():
                    print(f"Recognized: {query}")
                    self.process_query(query)
                else:
                    print(query or "Voice input failed.")
            elif choice == "3":
                path = input("Image path: ").strip()
                if path:
                    result = self.analyze_medical_image(path)
                    print(f"\nAnalysis: {result}")
                    self.memory.chat_memory.add_message(AIMessage(content=result))
                else:
                    print("Please provide an image path.")
            elif choice == "q":
                print("Goodbye.")
                break
            else:
                print("Invalid option. Choose 1, 2, 3, or q.")


# In[ ]:


if __name__ == "__main__":
    assistant = MedicalAssistant()
    assistant.run()


# In[ ]:




