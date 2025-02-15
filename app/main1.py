import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pymongo import MongoClient
import stripe

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")

# Initialize OpenAI API
OpenAI.api_key = OPENAI_API_KEY

# Initialize Stripe
stripe.api_key = STRIPE_SECRET_KEY

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to MongoDB
client = MongoClient(MONGODB_URI)
db = client.grocerydb
users_collection = db.users

# Request model
class ShoppingListRequest(BaseModel):
    uid: str  # Firebase UID for user identification
    latitude: float
    longitude: float
    items: List[str]
    meal_preference: Optional[List[str]] = []  # Example: ["Asian", "Italian"]
    food_allergies: Optional[List[str]] = []   # Example: ["Eggs", "Dairy"]
    prefers_shopping_at: Optional[List[str]] = []  # Example: ["Target", "Walmart"]
    likes_to_eat: Optional[List[str]] = []  # Example: ["Xiaolongbao"]
    longest_distance: Optional[float] = 5  # Default to 5 miles

@app.get("/")
def read_root():
    return {"message": "Welcome to the Grocery Recommender System!"}

# Fetch user preferences from MongoDB
def get_user_preferences(uid: str):
    user = users_collection.find_one({"uid": uid})
    if not user:
        return None
    return {
        "meal_preference": user.get("meal_preference", []),
        "food_allergies": user.get("food_allergies", []),
        "prefers_shopping_at": user.get("prefers_shopping_at", []),
        "likes_to_eat": user.get("likes_to_eat", []),
        "longest_distance": user.get("longest_distance", 5)
    }

# Fetch nearby stores from Google Places API
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
                "center": {"latitude": lat, "longitude": lon},
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

# Recommend the cheapest store
@app.post("/recommend-cheapest-store")
def recommend_cheapest_store(request: ShoppingListRequest):
    lat, lon = request.latitude, request.longitude
    items = request.items
    uid = request.uid

    # Get user preferences from MongoDB
    user_prefs = get_user_preferences(uid)
    if not user_prefs:
        raise HTTPException(status_code=404, detail="User preferences not found")

    meal_pref = user_prefs["meal_preference"]
    allergies = set(user_prefs["food_allergies"])
    preferred_stores = user_prefs["prefers_shopping_at"]
    likes_to_eat = user_prefs["likes_to_eat"]
    longest_distance = user_prefs["longest_distance"]

    # Filter out allergy-related items
    filtered_items = [item for item in items if item.lower() not in allergies]

    try:
        nearby_data = fetch_nearby_stores(lat, lon, radius=longest_distance * 1609)  # Convert miles to meters
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch store data: {str(e)}")

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
    User Preferences:
    - UID: {uid}
    - Meal Preferences: {meal_pref}
    - Likes to Eat: {likes_to_eat}
    - Food Allergies: {allergies}
    - Preferred Stores: {preferred_stores}
    
    The user wants to buy the following items: {filtered_items}.
    They are located at ({lat}, {lon}).
    Here are some nearby stores:
    {stores_text}

    Suggest the best store considering price, distance, and user preferences.
    """

    client = OpenAI()
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a grocery assistant providing personalized recommendations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        recommendation_text = completion.choices[0].message.content.strip()
    except Exception as e:
        recommendation_text = f"Could not generate recommendation: {str(e)}"

    return {
        "filtered_shopping_list": filtered_items,
        "llm_recommendation": recommendation_text
    }

# Find store directions using Google Maps API
@app.post("/find-directions")
def find_directions(request: ShoppingListRequest):
    stores = fetch_nearby_stores(request.latitude, request.longitude, radius=request.longest_distance * 1609)
    waypoints = "|".join([store.get("formattedAddress", "") for store in stores.get("places", []) if "formattedAddress" in store])
    
    url = f"https://www.google.com/maps/dir/?api=1&origin={request.latitude},{request.longitude}&destination={waypoints}&travelmode=driving"
    
    return {"map_url": url, "stores": [store.get("displayName", "Unknown Store") for store in stores.get("places", [])]}

# Stripe checkout for groceries
@app.post("/checkout")
def checkout(uid: str, items: List[str], total_amount: float):
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price_data": {"currency": "usd", "product_data": {"name": item}, "unit_amount": int(total_amount * 100)}, "quantity": 1} for item in items],
            mode="payment",
            success_url="https://yourfrontend.com/success",
            cancel_url="https://yourfrontend.com/cancel"
        )
        return {"checkout_url": session.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating checkout session: {str(e)}")
