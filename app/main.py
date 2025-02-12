import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any


from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

OpenAI.api_key = OPENAI_API_KEY

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mongo_client = MongoClient(MONGODB_URI)
db = mongo_client.grocerydb
onboarding_collection = db.onboardingData 

from pydantic import BaseModel
from typing import Optional, List

class UserPreferences(BaseModel):
    uid: str
    cuisinePreferences: Optional[List[str]] = None
    foodAllergies: Optional[List[str]] = None
    dietaryPreferences: Optional[List[str]] = None
    preferences: Optional[Dict[str, Any]] = None 


class ShoppingListRequest(BaseModel):
    latitude: float
    longitude: float
    items: List[str]

@app.get("/")
def read_root():
    return {"message": "Hello from the Grocery Recommender (New Nearby Search)!"}

@app.post("/onboarding")
def update_user_preferences(data: UserPreferences):
    try:
        if not data.uid:
            raise HTTPException(status_code=400, detail="User ID (uid) is required")

        existing_user = onboarding_collection.find_one({"uid": data.uid})

        update_fields = {}

        if data.cuisinePreferences:
            updated_cuisines = set(data.cuisinePreferences)
            if existing_user and "cuisinePreferences" in existing_user:
                updated_cuisines.update(existing_user["cuisinePreferences"])
            update_fields["cuisinePreferences"] = list(updated_cuisines)

        if data.foodAllergies:
            updated_allergies = set(data.foodAllergies)
            if existing_user and "foodAllergies" in existing_user:
                updated_allergies.update(existing_user["foodAllergies"])
            update_fields["foodAllergies"] = list(updated_allergies)

        if data.dietaryPreferences:
            updated_dietary = set(data.dietaryPreferences)
            if existing_user and "dietaryPreferences" in existing_user:
                updated_dietary.update(existing_user["dietaryPreferences"])
            update_fields["dietaryPreferences"] = list(updated_dietary)

        if data.preferences:
            updated_preferences = existing_user.get("preferences", {}) if existing_user else {}
            for key, value in data.preferences.items():
                if isinstance(value, list): 
                    updated_preferences[key] = list(set(updated_preferences.get(key, []) + value))
                elif isinstance(value, dict):
                    updated_preferences[key] = {**updated_preferences.get(key, {}), **value}
                else:
                    updated_preferences[key] = value

            update_fields["preferences"] = updated_preferences 

        if not update_fields:
            raise HTTPException(status_code=400, detail="No valid fields provided for update")

        onboarding_collection.update_one(
            {"uid": data.uid},
            {"$set": update_fields},
            upsert=True
        )

        return {"message": "User preferences updated successfully", "updated_fields": update_fields}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    

@app.get("/onboarding/{uid}")
def get_user_preferences(uid: str):
    user_data = onboarding_collection.find_one({"uid": uid}, {"_id": 0})
    if not user_data:
        raise HTTPException(status_code=404, detail="User preferences not found")
    return user_data


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
        # "shopping_list": items,
        # "google_places_data": nearby_data, 
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


