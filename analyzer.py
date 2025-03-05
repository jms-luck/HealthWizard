import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import os
import shutil
import google.generativeai as genai
import cv2
from deep_translator import GoogleTranslator
import io
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
import json
import requests
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from googlesearch import search  # Import the googlesearch library
import uuid

# Set Tesseract path (adjust based on your system)

# Check if Tesseract is installed

# Set your Google API key directly
api_key = "AIzaSyAO-n2dhhke4Cq_Iix1-bILoU6EY7VNsnM" # Replace with your actual API key
GOOGLE_API_KEY= "AIzaSyAO-n2dhhke4Cq_Iix1-bILoU6EY7VNsnM" # Replace with your actual API key

# Configure the API client
genai.configure(api_key=api_key)

# Add Azure configuration after your Google API key setup
AZURE_ENDPOINT = "https://dscfdsdszsr.cognitiveservices.azure.com/"
AZURE_KEY = "Eni45CPxDqNDF24sabmmIaxIosLh0MkYIN69P4Xo5Qg6cGzmgBTEJQQJ99BCACYeBjFXJ3w3AAAFACOGBd8D"

# Initialize the Computer Vision client
vision_client = ComputerVisionClient(
    endpoint=AZURE_ENDPOINT,
    credentials=CognitiveServicesCredentials(AZURE_KEY)
)

# Update the Azure Translator configuration
TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
TRANSLATOR_KEY = "BUZB48JYJGc9PPdCudfMKNLPwLuj9zu4Oye4ZDTTDmEgLvpHLkQBJQQJ99BBACLArgHXJ3w3AAAbACOG2h3q"
TRANSLATOR_LOCATION = "eastus"  # Make sure this matches your Azure resource location


# ----- Normal Reference Ranges -----
NORMAL_RANGES = {
    "Hemoglobin": (13.0, 16.5),
    "RBC Count": (4.5, 5.5),
    "Hematocrit": (40, 49),
    "MCV": (83, 101),
    "MCH": (27.1, 32.5),
    "MCHC": (32.5, 36.7),
    "RDW CV": (11.6, 14),
    "WBC Count": (4000, 10000),
    "Platelet Count": (150000, 410000),
    "MPV": (7.5, 10.3),
    "ESR": (0, 14),
    "Cholesterol": (0, 200),
    "Triglyceride": (0, 150),
    "HDL Cholesterol": (40, 60),
    "Direct LDL": (0, 100),
    "VLDL": (15, 35),
    "CHOL/HDL Ratio": (0, 5.0),
    "LDL/HDL Ratio": (0, 3.5),
    "Fasting Blood Sugar": (74, 106),
    "HbA1c": (0, 5.7),
    "Mean Blood Glucose": (80, 120),
    "T3 - Triiodothyronine": (0.58, 1.59),
    "T4 - Thyroxine": (4.87, 11.72),
    "TSH - Thyroid Stimulating Hormone": (0.35, 4.94),
    "Microalbumin": (0, 16.7),
    "Total Protein": (6.3, 8.2),
    "Albumin": (3.5, 5.0),
    "Globulin": (2.3, 3.5),
    "A/G Ratio": (1.3, 1.7),
    "Total Bilirubin": (0.2, 1.3),
    "Conjugated Bilirubin": (0.0, 0.3),
    "Unconjugated Bilirubin": (0.0, 1.1),
    "Delta Bilirubin": (0.0, 0.2),
    "Iron": (49, 181),
    "Total Iron Binding Capacity": (261, 462),
    "Transferrin Saturation": (20, 50),
    "Homocysteine, Serum": (6.0, 14.8),
    "Creatinine, Serum": (0.66, 1.25),
    "Urea": (19.3, 43.0),
    "Blood Urea Nitrogen": (9.0, 20.0),
    "Uric Acid": (3.5, 8.5),
    "Calcium": (8.4, 10.2),
    "SGPT": (0, 50),
    "SGOT": (17, 59),
    "Sodium": (136, 145),
    "Potassium": (3.5, 5.1),
    "Chloride": (98, 107),
    "25(OH) Vitamin D": (30, 100),
    "Vitamin B12": (187, 833),
    "IgE": (0, 87),
    "PSA-Prostate Specific Antigen, Total": (0, 40.57)
}

# ----- Logical Groups for Visualizations -----
GROUPS = {
    "Complete Blood Count": ["Hemoglobin", "RBC Count", "Hematocrit", "MCV", "MCH", "MCHC", "RDW CV", "WBC Count", "Platelet Count", "MPV", "ESR"],
    "Lipid Profile": ["Cholesterol", "Triglyceride", "HDL Cholesterol", "Direct LDL", "VLDL", "CHOL/HDL Ratio", "LDL/HDL Ratio"],
    "Blood Sugar": ["Fasting Blood Sugar", "HbA1c", "Mean Blood Glucose"],
    "Thyroid": ["T3 - Triiodothyronine", "T4 - Thyroxine", "TSH - Thyroid Stimulating Hormone"],
    "Urine": ["Microalbumin", "Total Protein", "Albumin", "Globulin", "A/G Ratio", "Total Bilirubin", "Conjugated Bilirubin", "Unconjugated Bilirubin", "Delta Bilirubin"],
    "Iron Studies": ["Iron", "Total Iron Binding Capacity", "Transferrin Saturation"],
    "Renal": ["Creatinine, Serum", "Urea", "Blood Urea Nitrogen", "Uric Acid"],
    "Electrolytes": ["Calcium", "Sodium", "Potassium", "Chloride"],
    "Vitamins": ["25(OH) Vitamin D", "Vitamin B12"],
    "Other": ["IgE", "Homocysteine, Serum", "PSA-Prostate Specific Antigen, Total"]
}

# Add this dictionary after your GROUPS dictionary
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Russian": "ru",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu"
}

# ----- Functions -----

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])
        if not text.strip():
            raise ValueError("No text found in PDF.")
        return text.strip()
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")
        return ""

def extract_medical_values(text):
    """Extract medical parameter values from text."""
    extracted_data = {}
    for parameter, _ in NORMAL_RANGES.items():
        pattern = rf"{re.escape(parameter)}"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = match.start()
            window = text[start:start+150]
            numbers = re.findall(r"\d+(?:\.\d+)?", window)
            valid_numbers = [n for n in numbers if re.match(r"^\d+(?:\.\d+)?$", n)]
            if valid_numbers:
                try:
                    if len(valid_numbers) >= 3:
                        measured = float(valid_numbers[-1])
                    elif len(valid_numbers) == 2:
                        measured = float(valid_numbers[1])
                    else:
                        measured = float(valid_numbers[0])
                    extracted_data[parameter] = measured
                except ValueError:
                    st.warning(f"Could not convert value for {parameter} to float.")
    return extracted_data

def plot_medical_report(extracted_data, title="Medical Report Parameters vs Normal Ranges"):
    """Plot medical data against normal ranges."""
    if not extracted_data:
        st.warning("No medical values extracted.")
        return

    labels, values, low_lines, high_lines, colors = [], [], [], [], []
    for parameter, measured in extracted_data.items():
        if parameter in NORMAL_RANGES:
            low, high = NORMAL_RANGES[parameter]
            labels.append(parameter)
            values.append(measured)
            low_lines.append(low)
            high_lines.append(high)
            colors.append("blue" if measured < low else "red" if measured > high else "green")

    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels)*0.35)))
    ax.barh(y_pos, values, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Measured Value")
    ax.set_title(title)

    for i in range(len(labels)):
        ax.plot([low_lines[i], low_lines[i]], [i - 0.4, i + 0.4], "gray", linestyle="dashed", linewidth=1)
        ax.plot([high_lines[i], high_lines[i]], [i - 0.4, i + 0.4], "gray", linestyle="dashed", linewidth=1)
        ax.text(values[i], i, f" {values[i]}", va="center", ha="left", fontsize=8)

    st.pyplot(fig)

def extract_text_from_image(image_file, target_language="en"):
    """Extract text from an image using Azure Computer Vision or Tesseract as fallback."""
    try:
        # If Azure client is available, try Azure OCR first
        if vision_client:
            image_bytes = image_file.getvalue()
            read_response = vision_client.read_in_stream(io.BytesIO(image_bytes), raw=True)
            read_operation_location = read_response.headers["Operation-Location"]
            operation_id = read_operation_location.split("/")[-1]

            # Wait for the operation to complete
            while True:
                read_result = vision_client.get_read_result(operation_id)
                if read_result.status not in ['notStarted', 'running']:
                    break
                time.sleep(1)

            # Extract text from Azure results
            azure_text = ""
            if read_result.status == OperationStatusCodes.succeeded:
                for text_result in read_result.analyze_result.read_results:
                    for line in text_result.lines:
                        azure_text += line.text + "\n"
            
            # If Azure OCR fails, fall back to Tesseract
            if not azure_text.strip():
                # Reset file pointer and use PIL/Tesseract
                image_file.seek(0)
                image = Image.open(image_file)
                # Convert to array for OpenCV processing
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                # Apply image preprocessing
                processed_img = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 31, 2
                )
                # Extract text using Tesseract
                azure_text = pytesseract.image_to_string(processed_img)

        else:
            # Skip Azure and use Tesseract directly
            image_file.seek(0)
            image = Image.open(image_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            processed_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 31, 2
            )
            azure_text = pytesseract.image_to_string(processed_img)

        # Process extracted text
        med_pattern = r"(Tab|Syp|Cap|Inj|Rx)\.?\s([A-Za-z0-9\-]+)"
        medicines = re.findall(med_pattern, azure_text)
        
        # Format output
        final_text = (
            "EXTRACTED TEXT:\n" + 
            "-" * 50 + "\n" +
            azure_text + "\n\n" +
            "RECOGNIZED MEDICINES:\n" +
            "-" * 50 + "\n" +
            "\n".join([f"🩺 {med_type}. {med_name}" for med_type, med_name in medicines])  # Simplified medicine list
        )
        
        return final_text.strip(), medicines  # Return medicines list
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return "", []  # Return empty list in case of error

def analyze_handwritten_prescription(text, translate_to=None):
    """Analyze and complete handwritten prescription text using AI with optional translation."""
    if not GOOGLE_API_KEY:
        st.error("Google API key is not set. Please set the GOOGLE_API_KEY environment variable.")
        return "AI analysis failed due to missing API key."
    try:
        prompt = f"""
You are a medical assistant. Given the following handwritten prescription text, please:
1. Analyze the text and provide the probable medical condition being treated.
2. Suggest a complete prescription with clear usage instructions if any text seems incomplete or unclear.

Prescription Text:
{text}

Please ensure the text is patient-friendly, correcting any mistakes and completing missing parts where possible.
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        result = response.text if response and response.text else "AI analysis and completion failed. Please try again."
        
        # Translate result if a target language is specified
        if translate_to and result:
            try:
                path = '/translate'
                constructed_url = TRANSLATOR_ENDPOINT + path

                params = {
                    'api-version': '3.0',
                    'from': 'en',
                    'to': [translate_to]  # API expects an array
                }

                headers = {
                    'Ocp-Apim-Subscription-Key': TRANSLATOR_KEY,
                    'Ocp-Apim-Subscription-Region': TRANSLATOR_LOCATION,
                    'Content-type': 'application/json',
                    'X-ClientTraceId': str(uuid.uuid4())
                }

                # Debug logging
                with st.expander("Debug Info"):
                    st.write("Request URL:", constructed_url)
                    st.write("Request Headers:", {k: v for k, v in headers.items() if k != 'Ocp-Apim-Subscription-Key'})
                    st.write("Request Params:", params)

                body = [{
                    'text': result
                }]

                translation_response = requests.post(
                    constructed_url,
                    params=params,
                    headers=headers,
                    json=body,
                    timeout=10
                )

                if translation_response.status_code == 200:
                    translations = translation_response.json()
                    if translations and len(translations) > 0:
                        translated = translations[0]['translations'][0]['text']
                        return translated
                else:
                    st.error(f"Translation failed ({translation_response.status_code}): {translation_response.text}")
                    return result  # Fall back to original text
                    
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
                return result  # Fall back to original text
        
        return result
    except Exception as e:
        st.error(f"Error during AI analysis: {e}")
        return "AI analysis failed due to an error."

def search_online_pharmacies(medicine_name):
    """Search for specific online pharmacy websites for medicines."""
    pharmacy_urls = {
        "MedPlus": f"https://www.medplusmart.com/searchProduct?text={medicine_name}",
        "NetMeds": f"https://www.netmeds.com/catalogsearch/result/{medicine_name}/all",
        "1mg": f"https://www.1mg.com/search/all?name={medicine_name}",
        "PharmEasy": f"https://pharmeasy.in/search/all?name={medicine_name}",
        "Apollo Pharmacy": f"https://www.apollopharmacy.in/search-medicines/{medicine_name}"
    }
    return pharmacy_urls

def get_current_location():
    """Get user's approximate location using an IP-based geolocation service."""
    try:
        response = requests.get("https://ipinfo.io/json").json()
        loc = response["loc"].split(",")  # Extract latitude and longitude
        latitude, longitude = float(loc[0]), float(loc[1])
        return latitude, longitude
    except Exception as e:
        st.error("Error getting location: " + str(e))
        return None

def get_nearby_pharmacies(lat, lon, radius=5000):
    """Find nearby pharmacies using OpenStreetMap (Overpass API)."""
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node
      ["amenity"="pharmacy"]
      (around:{radius},{lat},{lon});
    out;
    """
    response = requests.get(overpass_url, params={"data": query})
    data = response.json()

    pharmacies = []
    for element in data.get("elements", []):
        name = element.get("tags", {}).get("name", "Unknown Pharmacy")
        lat, lon = element["lat"], element["lon"]
        pharmacies.append({"name": name, "latitude": lat, "longitude": lon})
    
    return pharmacies

def create_map(user_lat, user_lon, pharmacies):
    """Generate an interactive map with Folium."""
    map_ = folium.Map(location=[user_lat, user_lon], zoom_start=14)

    # Add user location marker
    folium.Marker([user_lat, user_lon], 
                  popup="Your Location", 
                  icon=folium.Icon(color="blue", icon="user")).add_to(map_)

    # Add pharmacy markers
    for pharmacy in pharmacies:
        folium.Marker(
            [pharmacy["latitude"], pharmacy["longitude"]],
            popup=pharmacy["name"],
            icon=folium.Icon(color="red", icon="plus-sign"),
        ).add_to(map_)

    return map_

# ----- Main App -----

def main():
    st.title("Medical Report Analyzer & Prescription Reader")
    analysis_type = st.sidebar.radio("Choose Analysis Type", ["Lab Report", "Handwritten Prescription"])

    if analysis_type == "Lab Report":
        st.write("Upload your lab report (PDF) to visualize parameters against normal ranges.")
        uploaded_file = st.file_uploader("Upload Lab Report (PDF)", type=["pdf"])
        if uploaded_file is not None:
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                st.error("No text could be extracted from the PDF.")
                return

            with st.expander("Extracted Text Preview"):
                st.text_area("Extracted Text", value=text, height=200)

            medical_values = extract_medical_values(text)
            if medical_values:
                st.subheader("Extracted Medical Values (Cumulative)")
                st.dataframe(pd.DataFrame(list(medical_values.items()), columns=["Parameter", "Measured Value"]))
                st.subheader("Cumulative Visualization")
                plot_medical_report(medical_values)
                st.subheader("Separate Component Visualizations")
                for group_name, parameters in GROUPS.items():
                    group_data = {p: medical_values[p] for p in parameters if p in medical_values}
                    if group_data:
                        st.markdown(f"{group_name}")
                        st.dataframe(pd.DataFrame(list(group_data.items()), columns=["Parameter", "Measured Value"]))
                        plot_medical_report(group_data, title=f"{group_name}: Parameters vs Normal Ranges")
                    else:
                        st.info(f"No data extracted for {group_name}.")
            else:
                st.warning("No medical parameters could be extracted from the report.")

    elif analysis_type == "Handwritten Prescription":
        st.write("Upload an image of a handwritten prescription.")
        
        # Add language selection
        target_language = st.selectbox(
            "Select Translation Language",
            options=list(LANGUAGE_CODES.keys()),
            index=0
        )
        
        image_file = st.file_uploader("Upload Prescription (Image)", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            ocr_text, medicines = extract_text_from_image(  # Capture medicines list
                image_file, 
                target_language="en" # Set default language
            )
            if not ocr_text:
                st.error("No text could be extracted from the image.")
            elif len(ocr_text) < 20:
                st.warning("Extracted text is very short. The image might not be clear.")
            else:
                with st.expander("Extracted Prescription Text"):
                    st.text_area("OCR Extracted Text", value=ocr_text, height=200)
                st.info("Analyzing handwritten prescription...")
                analysis_result = analyze_handwritten_prescription(ocr_text, translate_to=LANGUAGE_CODES[target_language])
                st.subheader("Handwritten Prescription Analysis")
                st.write(analysis_result)

                # --- Pharmacy Information ---
                if medicines:
                    st.subheader("🏪 Buy Medicines Online")
                    
                    # Create tabs for each medicine
                    med_tabs = st.tabs([f"{med_type}. {med_name}" for med_type, med_name in medicines])
                    
                    for i, (med_type, med_name) in enumerate(medicines):
                        with med_tabs[i]:
                            st.write(f"### {med_name}")
                            online_pharmacies = search_online_pharmacies(med_name)
                            
                            # Display pharmacy options in a grid
                            cols = st.columns(len(online_pharmacies))
                            for idx, (pharmacy_name, url) in enumerate(online_pharmacies.items()):
                                with cols[idx]:
                                    st.link_button(f"Buy on {pharmacy_name}", url)
                                    st.caption(f"Search {med_name} on {pharmacy_name}")

                    # --- Nearby Pharmacies Locator ---
                    st.markdown("---")
                    st.subheader("🏥 Nearby Physical Pharmacies")
                    location = get_current_location()
                    
                    # ... rest of the existing nearby pharmacies code ...
                    if location:
                        lat, lon = location
                        st.success(f"Your Current Location: {lat}, {lon}")

                        # Get nearby pharmacies
                        pharmacies = get_nearby_pharmacies(lat, lon)

                        if pharmacies:
                            # Show pharmacy list
                            st.subheader("Nearby Pharmacies")
                            df = gpd.GeoDataFrame(pharmacies, 
                                                  geometry=gpd.points_from_xy([p["longitude"] for p in pharmacies],
                                                                              [p["latitude"] for p in pharmacies]),
                                                  crs="EPSG:4326")
                            st.dataframe(df[["name", "latitude", "longitude"]])

                            # Show map
                            st.subheader("Pharmacy Locations on Map")
                            map_ = create_map(lat, lon, pharmacies)
                            folium_static(map_)
                        else:
                            st.warning("No pharmacies found nearby.")
                    else:
                        st.error("Could not retrieve your location.")
                else:
                    st.info("No medicines recognized in the prescription.")

if __name__ == "__main__":  # This is correct
    main()
