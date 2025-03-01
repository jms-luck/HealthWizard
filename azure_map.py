#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
import json


# In[3]:


# Configure your Azure Maps subscription key here.
subscription_key = "EumXcWSYqKLcsw9zymB1cPRIfDzNbZBXO7BCjKsbsAITXSpRIZbMJQQJ99BBACYeBjFPDDZUAAAgAZMP1DsH"  # <-- Replace with your Azure Maps key. If left empty, the script will prompt for it.

def search_location(query, subscription_key):
    """Search for a location using Azure Maps Search API (Fuzzy Search)."""
    # Azure Maps Search (Fuzzy) endpoint URL
    base_url = "https://atlas.microsoft.com/search/fuzzy/json"
    params = {
        "api-version": "1.0",
        "subscription-key": subscription_key,
        "query": query
    }
    try:
        # Make the HTTP GET request to Azure Maps Search API
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx/5xx)
    except requests.exceptions.RequestException as e:
        # Handle any network or HTTP errors
        print(f"Error during API request: {e}")
        return None

    # Parse the JSON response
    try:
        data = response.json()
    except ValueError:
        print("Error: Received response is not valid JSON.")
        return None

    return data

if __name__ == "__main__":
    # Prompt for subscription key if not provided
    if not subscription_key:
        subscription_key = input("Enter your Azure Maps subscription key: ").strip()
    if not subscription_key:
        print("No subscription key provided. Exiting.")
        exit(1)

    # Prompt user for a search query (location or address)
    query = input("Enter a location or address to search: ").strip()
    if not query:
        print("No search query provided. Exiting.")
        exit(1)

    # Perform the location search
    result_data = search_location(query, subscription_key)
    if result_data is None:
        # An error occurred during the API request; message already printed
        exit(1)

    # Print the full JSON result in a structured format
    print("\nFull JSON response:")
    print(json.dumps(result_data, indent=2))

    # Check if any results were returned
    results = result_data.get("results")
    if not results:
        print("\nNo results found for query:", query)
    else:
        # Display key details for each result
        print("\nSearch results:")
        for idx, res in enumerate(results, start=1):
            # Some results have a 'poi' field with the name for points of interest
            # If 'poi' is missing or has no name, use the freeform address as name
            name = None
            if "poi" in res and "name" in res["poi"]:
                name = res["poi"]["name"]
            elif "address" in res and "freeformAddress" in res["address"]:
                name = res["address"]["freeformAddress"]
            else:
                name = "(unknown name)"
            lat = res.get("position", {}).get("lat")
            lon = res.get("position", {}).get("lon")
            print(f"{idx}. Name: {name}")
            print(f"   Latitude: {lat}, Longitude: {lon}")


# In[ ]:




