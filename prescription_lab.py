#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from googletrans import Translator
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
from geopy.geocoders import Nominatim
from googlesearch import search  # Import the googlesearch library
import argparse


# In[ ]:





# In[ ]:


# ---------------- Configuration ----------------

# Set Tesseract path (adjust based on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if shutil.which("tesseract") is None:
    print("Error: Tesseract is not installed or not in your PATH. Please install Tesseract OCR and add it to your PATH.")
    exit(1)

# Google API configuration
api_key = "AIzaSyAO-n2dhhke4Cq_Iix1-bILoU6EY7VNsnM"  # Replace with your actual API key
GOOGLE_API_KEY = api_key

# Configure the API client for Generative AI
genai.configure(api_key=api_key)

# Azure Computer Vision configuration
AZURE_ENDPOINT = "https://hackocr.cognitiveservices.azure.com/"
AZURE_KEY = "16vHEtnFtkwA2ZJlKEAG3MUPwCW0DOrwCrxzituErLU5F6OD2tTuJQQJ99BBACYeBjFXJ3w3AAAFACOGILIl"

vision_client = ComputerVisionClient(
    endpoint=AZURE_ENDPOINT,
    credentials=CognitiveServicesCredentials(AZURE_KEY)
)

# Azure Translator configuration
TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
TRANSLATOR_KEY = "BUZB48JYJGc9PPdCudfMKNLPwLuj9zu4Oye4ZDTTDmEgLvpHLkQBJQQJ99BBACLArgHXJ3w3AAAbACOG2h3q"

translator_credential = AzureKeyCredential(TRANSLATOR_KEY)
translator_client = TextTranslationClient(
    endpoint=TRANSLATOR_ENDPOINT, 
    credential=translator_credential
)

# ---------------- Normal Reference Ranges ----------------

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

# ---------------- Logical Groups for Visualizations ----------------

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

# ---------------- Functions ----------------

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])
        if not text.strip():
            raise ValueError("No text found in PDF.")
        return text.strip()
    except Exception as e:
        print(f"Error processing PDF: {e}")
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
                    print(f"Warning: Could not convert value for {parameter} to float.")
    return extracted_data

def plot_medical_report(extracted_data, title="Medical Report Parameters vs Normal Ranges"):
    """Plot medical data against normal ranges."""
    if not extracted_data:
        print("Warning: No medical values extracted.")
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

    plt.tight_layout()
    plt.show()

def extract_text_from_image(image_file, target_language="en"):
    """Extract text from an image using Azure Computer Vision and Tesseract as fallback."""
    try:
        # Convert uploaded file to bytes
        image_bytes = image_file.read()
        
        # Call Azure's OCR with bytes
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
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            processed_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 31, 2
            )
            azure_text = pytesseract.image_to_string(processed_img)

        # Process extracted text to find medicines
        med_pattern = r"(Tab|Syp|Cap|Inj|Rx)\.?\s([A-Za-z0-9\-]+)"
        medicines = re.findall(med_pattern, azure_text)
        
        # Process medicines with Augmenti Translator
        medicine_details = []
        for med_type, med_name in medicines:
            try:
                augmenti_api_url = "https://api.augmenti.com/translate"
                headers = {
                    "Authorization": "Bearer BUZB48JYJGc9PPdCudfMKNLPwLuj9zu4Oye4ZDTTDmEgLvpHLkQBJQQJ99BBACLArgHXJ3w3AAAbACOG2h3q",
                    "Content-Type": "application/json"
                }
                payload = {
                    "text": med_name,
                    "targetLanguage": target_language
                }
                response = None
                try:
                    response = requests.post(augmenti_api_url, headers=headers, data=json.dumps(payload))
                    response.raise_for_status()
                except requests.exceptions.RequestException as req_err:
                    print(f"Warning: Augmenti API request failed for {med_name}: {str(req_err)}")
                    medicine_details.append(f"🩺 {med_type}. {med_name}")
                    continue

                if response is not None:
                    try:
                        translation_data = response.json()
                        if "translation" in translation_data:
                            translation = translation_data["translation"]
                            medicine_details.append(f"🩺 {med_type}. {med_name} → {translation}")
                        else:
                            medicine_details.append(f"🩺 {med_type}. {med_name} (Translation failed)")
                    except (ValueError, KeyError) as json_err:
                        print(f"Warning: Augmenti API response parsing failed for {med_name}: {str(json_err)}")
                        medicine_details.append(f"🩺 {med_type}. {med_name}")
                else:
                    medicine_details.append(f"🩺 {med_type}. {med_name} (Request failed)")

            except requests.exceptions.RequestException as req_err:
                print(f"Warning: Augmenti API request failed for {med_name}: {str(req_err)}")
                medicine_details.append(f"🩺 {med_type}. {med_name}")
            except (ValueError, KeyError) as json_err:
                print(f"Warning: Augmenti API response parsing failed for {med_name}: {str(json_err)}")
                medicine_details.append(f"🩺 {med_type}. {med_name}")

        final_text = (
            "EXTRACTED TEXT:\n" + "-" * 50 + "\n" +
            azure_text + "\n\n" +
            "RECOGNIZED MEDICINES:\n" + "-" * 50 + "\n" +
            "\n".join(medicine_details)
        )
        
        return final_text.strip(), medicines
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return "", []

def analyze_handwritten_prescription(text):
    """Analyze and complete handwritten prescription text using AI."""
    if not GOOGLE_API_KEY:
        print("Error: Google API key is not set.")
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
        return response.text if response and response.text else "AI analysis and completion failed. Please try again."
    except Exception as e:
        print(f"Error during AI analysis: {e}")
        return "AI analysis failed due to an error."

def search_online_pharmacies(medicine_name):
    """Search for online pharmacies selling the given medicine."""
    try:
        search_results = []
        for j in search(f"buy {medicine_name} online", stop=5, pause=2):
            search_results.append(j)
        return search_results
    except Exception as e:
        print(f"Warning: Error searching for {medicine_name}: {e}")
        return []

def get_current_location():
    """Get user's approximate location using an IP-based geolocation service."""
    try:
        response = requests.get("https://ipinfo.io/json").json()
        loc = response["loc"].split(",")
        latitude, longitude = float(loc[0]), float(loc[1])
        return latitude, longitude
    except Exception as e:
        print("Error getting location:", e)
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
        lat_elem, lon_elem = element["lat"], element["lon"]
        pharmacies.append({"name": name, "latitude": lat_elem, "longitude": lon_elem})
    
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

# ---------------- Main Application ----------------

def main():
    parser = argparse.ArgumentParser(description="Medical Report Analyzer & Prescription Reader")
    parser.add_argument("analysis_type", choices=["lab", "prescription"],
                        help="Type of analysis: 'lab' for Lab Report (PDF) or 'prescription' for Handwritten Prescription (Image)")
    parser.add_argument("file_path", help="Path to the file (PDF for lab or image for prescription)")
    parser.add_argument("--language", default="English", choices=list(LANGUAGE_CODES.keys()),
                        help="Target language for translation in prescription analysis (default: English)")
    args = parser.parse_args()

    if args.analysis_type == "lab":
        # Process Lab Report
        try:
            with open(args.file_path, "rb") as f:
                text = extract_text_from_pdf(f)
        except Exception as e:
            print(f"Error opening file: {e}")
            return

        if not text:
            print("No text could be extracted from the PDF.")
            return

        print("\n------ Extracted Text Preview ------\n")
        print(text[:1000])  # print first 1000 characters as a preview

        medical_values = extract_medical_values(text)
        if medical_values:
            print("\n------ Extracted Medical Values (Cumulative) ------")
            df = pd.DataFrame(list(medical_values.items()), columns=["Parameter", "Measured Value"])
            print(df.to_string(index=False))
            print("\n------ Cumulative Visualization ------")
            plot_medical_report(medical_values)

            print("\n------ Separate Component Visualizations ------")
            for group_name, parameters in GROUPS.items():
                group_data = {p: medical_values[p] for p in parameters if p in medical_values}
                if group_data:
                    print(f"\n{group_name}:")
                    df_group = pd.DataFrame(list(group_data.items()), columns=["Parameter", "Measured Value"])
                    print(df_group.to_string(index=False))
                    plot_medical_report(group_data, title=f"{group_name}: Parameters vs Normal Ranges")
                else:
                    print(f"No data extracted for {group_name}.")
        else:
            print("No medical parameters could be extracted from the report.")

    elif args.analysis_type == "prescription":
        # Process Handwritten Prescription
        try:
            with open(args.file_path, "rb") as f:
                ocr_text, medicines = extract_text_from_image(f, target_language=LANGUAGE_CODES[args.language])
        except Exception as e:
            print(f"Error opening file: {e}")
            return

        if not ocr_text:
            print("No text could be extracted from the image.")
        elif len(ocr_text) < 20:
            print("Warning: Extracted text is very short. The image might not be clear.")
        else:
            print("\n------ Extracted Prescription Text ------\n")
            print(ocr_text)
            print("\nAnalyzing handwritten prescription...\n")
            analysis_result = analyze_handwritten_prescription(ocr_text)
            print("\n------ Handwritten Prescription Analysis ------\n")
            print(analysis_result)

            if medicines:
                print("\n------ Recognized Medicines ------\n")
                for med_type, med_name in medicines:
                    print(f"{med_type}. {med_name}")

                # Nearby Pharmacies Locator
                print("\n------ Nearby Pharmacies ------")
                location = get_current_location()
                if location:
                    lat, lon = location
                    print(f"Your Current Location: {lat}, {lon}")

                    pharmacies = get_nearby_pharmacies(lat, lon)
                    if pharmacies:
                        df_pharmacies = pd.DataFrame(pharmacies)
                        print("\nNearby Pharmacies:")
                        print(df_pharmacies.to_string(index=False))
                        
                        # Create and save map
                        map_ = create_map(lat, lon, pharmacies)
                        map_file = "pharmacies_map.html"
                        map_.save(map_file)
                        print(f"\nPharmacy map has been saved to '{map_file}'. Open it in your browser to view the locations.")
                    else:
                        print("No pharmacies found nearby.")
                else:
                    print("Could not retrieve your location.")
            else:
                print("No medicines recognized in the prescription.")

if __name__ == "__main__":
    main()

