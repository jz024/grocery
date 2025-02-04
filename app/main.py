import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from openai import OpenAI

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

OpenAI.api_key = OPENAI_API_KEY

app = FastAPI()

# client = MongoClient(MONGODB_URI)
# db = client.grocerydb
# stores_collection = db.stores

# try:
#     print(client.list_database_names())
# except Exception as e:
#     print("Connection error:", e)

from pydantic import BaseModel
from typing import Optional, List


class ShoppingListRequest(BaseModel):
    latitude: float
    longitude: float
    items: List[str]

@app.get("/")
def read_root():
    return {"message": "Hello from the Grocery Recommender (New Nearby Search)!"}


@app.post("/recommend-cheapest-store")
def recommend_cheapest_store(request: ShoppingListRequest):
    lat, lon = request.latitude, request.longitude
    items = request.items

    try:
        nearby_data = fetch_nearby_stores(lat, lon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Google Places data: {str(e)}")
    
    store_summaries = []
    for place in nearby_data.get("places", []):
        store_name = place.get("displayName", "Unknown Store")
        formatted_address = place.get("formattedAddress", "Address not available")
        website_uri = place.get("websiteUri", "Website not available")
        location = place.get("location", {})
        lat_field = location.get("latitude", "N/A")
        lon_field = location.get("longitude", "N/A")

        store_summaries.append(
            f"- Name: {store_name}\n  Address: {formatted_address}\n  Website: {website_uri}\n  Location: Latitude {lat_field}, Longitude {lon_field}"
        )

    stores_text = "\n".join(store_summaries)


    # nearby_data = fetch_nearby_stores(lat, lon, query)


    # if nearby_data and "places" in nearby_data:
    #     for place in nearby_data["places"]:
    #         display_info = place.get("displayName", {})
    #         store_name = display_info.get("text", "Unknown Store")
    #         store_loc = place.get("location", {})
    #         lat_field = store_loc.get("latitude")
    #         lon_field = store_loc.get("longitude")
    #         place_id = place.get("name", "N/A")

    #         if not place_id:
    #             continue

    #         store_doc = {
    #             "place_id": place_id,
    #             "name": store_name,
    #             "location": {
    #                 "type": "Point",
    #                 "coordinates": [lon_field, lat_field] if (lon_field and lat_field) else []
    #             },
    #         }

    #         existing_store = stores_collection.find_one({"place_id": place_id})
    #         if not existing_store:
    #             stores_collection.insert_one(store_doc)

    # store_summaries = []
    # if nearby_data and "places" in nearby_data:
    #     for place in nearby_data["places"]:
    #         name_obj = place.get("displayName", {})
    #         store_name = name_obj.get("text", "Unknown Store")
    #         store_summaries.append(f"- {store_name}")

    # stores_text = "\n".join(store_summaries)

    prompt = f"""
    You are a helpful grocery assistant. 
    The user wants to buy the following items: {items}.
    They are located at ({lat}, {lon}).
    Here are some nearby stores:
    {stores_text}

    Identify which store is both cheapest and nearest for these items, 
    and explain your reasoning succinctly. If there's insufficient data, 
    please make the best guess.
    """

    client = OpenAI()

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful grocery assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        recommendation_text = completion.choices[0].message.content.strip()
    except Exception as e:
        recommendation_text = f"Could not generate recommendation: {str(e)}"

    print(recommendation_text)

    return {
        "shopping_list": items,
        "google_places_data": nearby_data, 
        "llm_recommendation": recommendation_text
    }


def fetch_nearby_stores(lat: float, lon: float, radius: float = 5000):
    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.websiteUri,places.location"
    }

    payload = {
        "includedTypes": ["grocery_store", "asian_grocery_store", "food_store", "supermarket"],
        "maxResultCount": 20,
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": lat,
                    "longitude": lon
                },
                "radius": float(radius)
            }
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching Google Places data: {response.json().get('error', {}).get('message', 'Unknown error')}"
        )
    return response.json()


