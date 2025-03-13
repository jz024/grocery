import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import requests
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pymongo import MongoClient
from fastapi.staticfiles import StaticFiles
import datetime
import json
import firebase_admin
from firebase_admin import credentials, firestore
import stripe
from motor.motor_asyncio import AsyncIOMotorClient
import httpx
import uuid

load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")

OpenAI.api_key = OPENAI_API_KEY
stripe.api_key = STRIPE_SECRET_KEY

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",
        "http://localhost:3000",
        "https://d375-2601-646-8f80-1110-4c4c-830b-652-52a4.ngrok-free.app",
        "https://accepted-flounder-supreme.ngrok-free.app",
        "*"  # Only use during development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# MongoDB Connection
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client.grocerydb
onboarding_collection = db.onboardingData
chat_collection = db.chatHistory

# Initialize Firebase Admin
cred = credentials.Certificate(os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH"))
firebase_admin.initialize_app(cred)
db_firebase = firestore.client()

# chat input schema
class ChatRequest(BaseModel):
    uid: str
    message: str

# Updated User Preferences Schema
class UserPreferences(BaseModel):
    uid: str
    cuisinePreferences: Optional[List[str]] = None
    foodAllergies: Optional[List[str]] = None
    dietaryPreferences: Optional[List[str]] = None
    preferences: Optional[Dict[str, Any]] = None 

# New CheckoutItem model
class CheckoutItem(BaseModel):
    name: str
    price: float
    quantity: int

# New CheckoutRequest model
class CheckoutRequest(BaseModel):
    items: List[CheckoutItem]

# New ImageSearchRequest model
class ImageSearchRequest(BaseModel):
    query: str

# Add these new models
class Ingredient(BaseModel):
    id: str
    name: str
    quantity: Optional[str] = None

class DietaryInfo(BaseModel):
    isVegetarian: bool
    isVegan: bool
    isGlutenFree: bool
    isDairyFree: bool

class MenuItem(BaseModel):
    id: Optional[str] = None
    name: str
    description: str
    image: Optional[str] = None
    ingredients: List[Ingredient]
    dietaryInfo: DietaryInfo
    rating: float

# Update the GenerateMenuRequest model
class GenerateMenuRequest(BaseModel):
    uid: str
    cuisine_type: Optional[str] = None
    dietary_preferences: Optional[List[str]] = None
    excluded_ingredients: Optional[List[str]] = None
    included_ingredients: Optional[List[str]] = None  # New field
    meal_count: Optional[int] = 10  # Default to 10 meals

# Add new models for categories
class Subcategory(BaseModel):
    id: str
    name: str

class Category(BaseModel):
    id: str
    name: str
    image: Optional[str] = None
    subcategories: List[str]

class Product(BaseModel):
    id: str
    name: str
    price: float
    image: Optional[str] = None
    description: str
    category: str
    subcategory: str

@app.post("/onboarding")
async def update_user_preferences(data: UserPreferences):
    try:
        if not data.uid:
            raise HTTPException(status_code=400, detail="User ID (uid) is required")

        existing_user = await onboarding_collection.find_one({"uid": data.uid})

        update_fields = {}

        # Merge lists
        def merge_lists(field_name, new_values):
            if new_values:
                updated_values = set(new_values)
                if existing_user and field_name in existing_user:
                    updated_values.update(existing_user[field_name])
                update_fields[field_name] = list(updated_values)

        merge_lists("cuisinePreferences", data.cuisinePreferences)
        merge_lists("foodAllergies", data.foodAllergies)
        merge_lists("dietaryPreferences", data.dietaryPreferences)

        # Update dictionary preferences
        if data.preferences:
            update_fields["preferences"] = {**existing_user.get("preferences", {}), **data.preferences}

        if not update_fields:
            raise HTTPException(status_code=400, detail="No valid fields provided for update")

        await onboarding_collection.update_one(
            {"uid": data.uid},
            {"$set": update_fields},
            upsert=True
        )

        return {"message": "User preferences updated successfully", "updated_fields": update_fields}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/onboarding/{uid}")
async def get_user_preferences(uid: str):
    user_data = await onboarding_collection.find_one({"uid": uid}, {"_id": 0})
    if not user_data:
        raise HTTPException(status_code=404, detail="User preferences not found")
    return user_data

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        uid = request.uid
        message = request.message

        # Fetch user preferences from Firebase (synchronous operation)
        user_ref = db_firebase.collection('users').document(uid).collection('AllAboutUser').document('preferences')
        user_doc = user_ref.get()  # This is synchronous
        
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User preferences not found")
        
        user_data = user_doc.to_dict()
        user_data = {
            "cuisinePreferences": user_data.get("cuisinePreferences", []),
            "foodAllergies": user_data.get("foodAllergies", []),
            "dietaryPreferences": user_data.get("dietaryPreferences", []),
            "location": user_data.get("location", "San Francisco, CA"),
        }

        # Fetch recent history (both chats and shopping lists)
        cursor = chat_collection.find(
            {
                "uid": uid,
                "type": {"$in": ["chat", "shopping_list"]}  # Include both types
            },
            {"_id": 0, "type": 1, "user_message": 1, "ai_response": 1, "list_data": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(5)  # Increased limit to include more context
        
        history = []
        async for doc in cursor:
            history.append(doc)

        # Reverse to get chronological order
        history.reverse()

        # Construct messages array with system prompt
        messages = [
            {
                "role": "system", 
                "content": """You are a helpful grocery assistant. Provide specific and practical advice about groceries, 
                recipes, and shopping while considering the user's preferences and restrictions. You have access to the 
                user's past messages and shopping lists. Maintain context of the conversation and reference previous 
                shopping lists when relevant."""
            }
        ]

        # Add user preferences context
        messages.append({
            "role": "system",
            "content": f"""User Preferences:
            - Cuisine Preferences: {', '.join(user_data['cuisinePreferences'])}
            - Food Allergies: {', '.join(user_data['foodAllergies'])}
            - Dietary Preferences: {', '.join(user_data['dietaryPreferences'])}
            - Location: {user_data['location']}"""
        })

        # Add history to messages
        for item in history:
            if item["type"] == "chat":
                if "user_message" in item:
                    messages.append({"role": "user", "content": item["user_message"]})
                if "ai_response" in item:
                    messages.append({"role": "assistant", "content": item["ai_response"]})
            elif item["type"] == "shopping_list":
                # Format shopping list as a clear message
                if "list_data" in item:
                    list_data = item["list_data"]
                    shopping_list_msg = "Previously generated shopping list:\n"
                    
                    if "locations" in list_data:
                        shopping_list_msg += f"Stores: {', '.join(list_data['locations'])}\n"
                    
                    if "items" in list_data:
                        shopping_list_msg += "Items:\n"
                        for item_entry in list_data["items"]:
                            shopping_list_msg += f"- {item_entry['name']} (${item_entry['price']:.2f} at {item_entry.get('location', 'any store')})\n"
                    
                    messages.append({"role": "assistant", "content": shopping_list_msg})

        # Add current message
        messages.append({"role": "user", "content": message})

        # Print messages for debugging
        print("Sending messages to OpenAI:", json.dumps(messages, indent=2))

        client = OpenAI()
        # Handle streaming response
        completion = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            stream=True
        )

        # Collect the streamed response
        ai_response = ""
        for chunk in completion:
            if hasattr(chunk.choices[0].delta, 'content'):
                if chunk.choices[0].delta.content is not None:
                    ai_response += chunk.choices[0].delta.content

        ai_response = ai_response.strip()

        # Save chat history
        await chat_collection.insert_one({
            "uid": uid,
            "type": "chat",
            "user_message": message,
            "ai_response": ai_response,
            "timestamp": datetime.datetime.now()
        })

        return {"ai_response": ai_response}

    except Exception as e:
        print(f"Chat error: {str(e)}")  # Log the full error
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/directions")
async def get_directions(locations: List[str]):
    try:
        if len(locations) < 2:
            raise HTTPException(status_code=400, detail="At least two locations are required")

        base_url = "https://maps.googleapis.com/maps/api/directions/json"
        waypoints = "|".join(locations[1:-1])

        async with httpx.AsyncClient() as client:
            response = await client.get(base_url, params={
                "origin": locations[0],
                "destination": locations[-1],
                "waypoints": waypoints,
                "key": GOOGLE_MAPS_API_KEY,
            })

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Error fetching directions from Google Maps API")

        return response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch directions: {str(e)}")

@app.post("/checkout")
def checkout(uid: str):
    try:
        # Fetch user shopping preferences
        chat_history = list(chat_collection.find({"uid": uid}, {"_id": 0}))

        if not chat_history:
            raise HTTPException(status_code=400, detail="No shopping history found for user")

        # Final AI confirmation
        prompt = f"""
        User has finished shopping. Here is their conversation history:
        {chat_history}

        AI: Confirm the items the user has decided to purchase.
        """

        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.5
        )

        confirmation_text = completion.choices[0].message.content.strip()

        return {"checkout_summary": confirmation_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process checkout: {str(e)}")

@app.get("/generate-shopping-list/{uid}")
async def generate_shopping_list(uid: str):
    try:
        # Fetch recent chat history
        cursor = chat_collection.find(
            {
                "uid": uid,
                "type": "chat"
            },
            {"_id": 0, "user_message": 1, "ai_response": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(10)
        
        chat_history = []
        async for doc in cursor:
            chat_history.append(doc)

        if not chat_history:
            return {"locations": [], "items": []}

        # Get user preferences from Firebase
        user_ref = db_firebase.collection('users').document(uid).collection('AllAboutUser').document('preferences')
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return {"items": [], "error": "User preferences not found"}
        
        user_prefs = user_doc.to_dict()
        
        # Construct the prompt
        chat_context = "\n".join([
            f"User: {chat.get('user_message', '')}\nAssistant: {chat.get('ai_response', '')}"
            for chat in reversed(chat_history)
        ])

        prompt = f"""
        Based on this conversation history and user preferences, generate a shopping list.
        
        User Preferences:
        - Cuisine Preferences: {', '.join(user_prefs.get('cuisinePreferences', []))}
        - Food Allergies: {', '.join(user_prefs.get('foodAllergies', []))}
        - Dietary Preferences: {', '.join(user_prefs.get('dietaryPreferences', []))}
        - Location: {user_prefs.get('location', 'Unknown')}

        Recent Conversation:
        {chat_context}

        Generate a shopping list in this exact JSON format:
        {{
            "items": [
                {{
                    "name": "Item name",
                    "price": estimated_price_in_dollars
                }}
            ]
        }}

        Requirements:
        1. Include all necessary ingredients mentioned in the conversation
        2. Respect dietary restrictions and allergies
        3. Use realistic price estimates
        4. Group similar items together
        5. Return only the JSON object, no additional text
        """

        print("\nGenerating shopping list...")
        print(f"User ID: {uid}")
        print(f"Chat history length: {len(chat_history)}")

        # Generate list using OpenAI
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are a JSON-generating assistant that creates shopping lists."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )

        # Get the response and clean it
        response_text = completion.choices[0].message.content.strip()
        print("\nReceived response from OpenAI")
        
        try:
            # Clean the response by removing markdown formatting
            if '```json' in response_text:
                # Extract the JSON content between markdown tags
                start_idx = response_text.find('```json') + 7
                end_idx = response_text.rfind('```')
                if end_idx == -1:
                    end_idx = len(response_text)
                response_text = response_text[start_idx:end_idx].strip()
            
            print("\nCleaned response:")
            print(response_text)
            
            shopping_list = json.loads(response_text)
            print(f"Successfully parsed shopping list with {len(shopping_list.get('items', []))} items")
            
            # Validate and clean shopping list
            if not isinstance(shopping_list, dict):
                print("Response is not a dictionary")
                return {"items": []}
            
            shopping_list.setdefault("items", [])
            
            if not isinstance(shopping_list["items"], list):
                print("Items is not a list")
                shopping_list["items"] = []
            
            valid_items = []
            for item in shopping_list["items"]:
                if isinstance(item, dict) and "name" in item and "price" in item:
                    if isinstance(item["name"], str) and isinstance(item["price"], (int, float)):
                        valid_items.append(item)
                        print(f"Added valid item: {item['name']}")
            
            shopping_list["items"] = valid_items
            print("Validated shopping list items")

            # Get store recommendations
            print("Getting store recommendations...")
            store_recommendations = await find_stores_for_items(uid, valid_items)
            
            # Create final document for MongoDB
            mongo_doc = {
                "uid": uid,
                "type": "shopping_list",
                "list_data": {
                    "items": shopping_list["items"],
                    "store_recommendations": store_recommendations if store_recommendations else {}
                },
                "timestamp": datetime.datetime.now()
            }

            # Save to MongoDB
            print("Saving to MongoDB...")
            await chat_collection.insert_one(mongo_doc)
            print("Shopping list saved successfully")

            return {
                "items": shopping_list["items"],
                "store_recommendations": store_recommendations if store_recommendations else {}
            }

        except json.JSONDecodeError as e:
            print(f"Failed to parse shopping list JSON: {str(e)}")
            print(f"Raw response: {response_text}")
            return {"items": []}

    except Exception as e:
        print(f"Shopping list generation error: {str(e)}")
        return {"items": []}

async def find_stores_for_items(uid: str, items: list) -> dict:
    """Helper function to find stores where items can be purchased"""
    try:
        # Get user location
        user_ref = db_firebase.collection('users').document(uid).collection('AllAboutUser').document('preferences')
        user_doc = user_ref.get()
        
        if not user_doc.exists or not user_doc.to_dict().get("location"):
            return {}

        location = user_doc.to_dict()["location"]
        items_text = ", ".join([item["name"] for item in items])
        
        # Construct query for stores
        search_query = f"""
        Find stores in {location} that sell these items: {items_text}
        
        Return the answer in this exact JSON format:
        {{
            "stores": [
                {{
                    "name": "Store name",
                    "address": "Full store address with city and zip code",
                    "available_items": ["item1", "item2"]
                }}
            ]
        }}
        """

        # Make Perplexity API call
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a store finder that returns only JSON data."
                },
                {
                    "role": "user",
                    "content": search_query
                }
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
            "Content-Type": "application/json"
        }

        # Use httpx for async HTTP requests
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            
            if response.status_code != 200:
                return {}

            response_data = response.json()
            store_data = response_data["choices"][0]["message"]["content"]
            
            # Clean and parse store data
            store_data = store_data.replace('```json\n', '').replace('\n```', '').strip()
            
            try:
                return json.loads(store_data)
            except json.JSONDecodeError:
                return {}

        return {}

    except Exception as e:
        print(f"Error finding stores: {str(e)}")
        return {}

@app.get("/test-store-search/{uid}")
async def test_store_search(uid: str):
    try:
        # Get the most recent shopping list
        last_shopping_list = chat_collection.find_one(
            {
                "uid": uid,
                "type": "shopping_list"
            },
            sort=[("timestamp", -1)]
        )

        if not last_shopping_list:
            raise HTTPException(
                status_code=404,
                detail="No shopping list found. Please generate a shopping list first."
            )

        # Get items from the shopping list
        items = [item["name"] for item in last_shopping_list["list_data"]["items"]]
        if not items:
            raise HTTPException(status_code=400, detail="Shopping list is empty")

        # Get user location
        user_ref = db_firebase.collection('users').document(uid).collection('AllAboutUser').document('preferences')
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            raise HTTPException(
                status_code=404,
                detail="User preferences not found. Please set your location in preferences."
            )
        
        user_data = user_doc.to_dict()
        location = user_data.get("location")
        
        if not location:
            raise HTTPException(
                status_code=400,
                detail="Location not set in preferences. Please set your location first."
            )

        # Construct query for all items
        items_text = ", ".join(items)
        search_query = f"""
        Find stores in {location} that sell these items: {items_text}
        
        Return the answer in this exact JSON format:
        {{
            "stores": [
                {{
                    "name": "Store name",
                    "address": "Full store address with city and zip code",
                    "available_items": ["item1", "item2"]
                }}
            ]
        }}
        Requirements:
        1. Only include real stores that actually exist at this location: {location}
        2. Include complete store addresses with zip codes
        3. Group items by store where they're most likely to be found
        4. Only include major grocery stores and specialty stores that definitely exist at this location
        Do not include any other text.
        """

        # Make Perplexity API call
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a store finder that returns only JSON data about stores and their available items."
                },
                {
                    "role": "user",
                    "content": search_query
                }
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
            "Content-Type": "application/json"
        }

        print("Making API call to Perplexity...")
        print("Searching for items:", items_text)
        response = requests.post(url, json=payload, headers=headers)
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")

        if not response.ok:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Perplexity API error: {response.text}"
            )

        response_data = response.json()
        store_data = response_data["choices"][0]["message"]["content"]
        
        # Clean up the response - remove markdown code blocks if present
        store_data = store_data.replace('```json\n', '').replace('\n```', '').strip()
        
        # If the response is a string, parse it as JSON
        if isinstance(store_data, str):
            try:
                store_data = json.loads(store_data)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {str(e)}")
                print(f"Raw store_data: {store_data}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to parse store recommendations"
                )

        # Validate the response structure
        if not isinstance(store_data, dict) or "stores" not in store_data:
            raise HTTPException(
                status_code=500,
                detail="Invalid response format from Perplexity API"
            )

        # Save the results
        await chat_collection.insert_one({
            "uid": uid,
            "type": "store_search",
            "original_items": items,
            "store_recommendations": store_data,
            "timestamp": datetime.datetime.now()
        })

        return {
            "original_items": items,
            "store_recommendations": store_data
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def update_preferences_from_history(uid: str, chat_history: list) -> dict:
    """
    Update user preferences based on chat history and return current preferences
    """
    # First, get current preferences from Firebase
    user_ref = db_firebase.collection('users').document(uid).collection('AllAboutUser').document('preferences')
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        current_preferences = {
            "cuisinePreferences": [],
            "foodAllergies": [],
            "dietaryPreferences": []
        }
    else:
        current_preferences = user_doc.to_dict()

    # Analyze chat history for new preferences and removed preferences
    client = OpenAI()
    chat_text = "\n".join([
        f"User: {chat.get('user_message', '')}\nAssistant: {chat.get('ai_response', '')}"
        for chat in reversed(chat_history)
    ])

    preference_prompt = f"""
    Analyze this conversation and identify both new preferences and preferences the user no longer wants.
    Current preferences: {current_preferences}

    Conversation:
    {chat_text}

    Return a valid JSON object with two sections - new preferences to add and existing preferences to remove:
    {{
        "add": {{
            "cuisinePreferences": ["new cuisine 1", "new cuisine 2"],
            "foodAllergies": ["new allergy 1"],
            "dietaryPreferences": ["new diet 1"]
        }},
        "remove": {{
            "cuisinePreferences": ["cuisine to remove 1"],
            "foodAllergies": ["allergy to remove 1"],
            "dietaryPreferences": ["diet to remove 1"]
        }}
    }}
    Only include preferences that are clearly stated or implied in the conversation.
    Return empty arrays if no changes are found.
    """

    preference_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a preference analyzer that returns only JSON."},
            {"role": "user", "content": preference_prompt}
        ],
        max_tokens=500,
        temperature=0.3
    )

    try:
        # Clean the response by removing markdown code block markers if present
        raw_response = preference_completion.choices[0].message.content.strip()
        cleaned_response = raw_response.replace('```json', '').replace('```', '').strip()
        
        preference_changes = json.loads(cleaned_response)
        
        print("\n=== Preference Changes Detected ===")
        print(json.dumps(preference_changes, indent=2))
        
        # Process additions and removals
        for key in ['cuisinePreferences', 'foodAllergies', 'dietaryPreferences']:
            # Add new preferences
            if key in preference_changes['add'] and preference_changes['add'][key]:
                current_set = set(current_preferences.get(key, []))
                new_set = set(preference_changes['add'][key])
                current_preferences[key] = list(current_set.union(new_set))
            
            # Remove preferences
            if key in preference_changes['remove'] and preference_changes['remove'][key]:
                current_set = set(current_preferences.get(key, []))
                remove_set = set(preference_changes['remove'][key])
                current_preferences[key] = list(current_set.difference(remove_set))

        print("\n=== Updated Preferences ===")
        print(json.dumps(current_preferences, indent=2))

        # Update Firebase with merged preferences
        try:
            user_ref.set(current_preferences, merge=True)
            
            # Verify the update by reading back from Firebase
            updated_doc = user_ref.get()
            if updated_doc.exists:
                verified_data = updated_doc.to_dict()
                print("\n=== Verified Firebase Update ===")
                print(json.dumps(verified_data, indent=2))
                
                # Check if all preference changes were saved correctly
                for key in ['cuisinePreferences', 'foodAllergies', 'dietaryPreferences']:
                    if key in current_preferences:
                        expected_set = set(current_preferences[key])
                        actual_set = set(verified_data.get(key, []))
                        if expected_set != actual_set:
                            print(f"\nWarning: Mismatch in {key}")
                            print(f"Expected: {expected_set}")
                            print(f"Actual: {actual_set}")
            else:
                print("\n=== Error: Failed to verify Firebase update ===")
                
        except Exception as firebase_error:
            print(f"\n=== Error Updating Firebase ===")
            print(f"Error: {str(firebase_error)}")

    except json.JSONDecodeError as e:
        print("\n=== Error Parsing Preference Changes ===")
        print(f"Error: {str(e)}")
        print("Raw response:", preference_completion.choices[0].message.content.strip())

    return current_preferences

@app.get("/test-firebase/{uid}")
def test_firebase_connection(uid: str):
    try:
        # Fetch user preferences from Firebase using the correct path
        user_ref = db_firebase.collection('users').document(uid).collection('AllAboutUser').document('preferences')
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User preferences not found")
        
        # Get the raw data from Firebase
        raw_data = user_doc.to_dict()
        
        # Format the data for display
        formatted_data = {
            "raw_firebase_data": raw_data,
            "processed_data": {
                "cuisinePreferences": raw_data.get("cuisinePreferences", []),
                "foodAllergies": raw_data.get("foodAllergies", []),
                "dietaryPreferences": raw_data.get("dietaryPreferences", []),
                "location": raw_data.get("location", "San Francisco, CA")
            }
        }
        
        return formatted_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Firebase error: {str(e)}")

@app.post("/create-checkout-session")
async def create_checkout_session(request: CheckoutRequest):
    try:
        line_items = [{
            'price_data': {
                'currency': 'usd',
                'product_data': {
                    'name': item.name,
                },
                'unit_amount': int(item.price * 100),
            },
            'quantity': item.quantity,
        } for item in request.items]

        # Use synchronous Stripe API call
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=line_items,
            mode='payment',
            success_url='https://yourdomain.com/success',
            cancel_url='https://yourdomain.com/cancel',
        )

        return {"url": session.url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search-image")
async def search_image(request: ImageSearchRequest):
    """Search for an image using Google Custom Search API"""
    try:
        if not request.query:
            raise HTTPException(status_code=400, detail="Query field is required")

        API_KEY = os.getenv("GOOGLE_API_KEY")
        CX = os.getenv("GOOGLE_CX")

        if not API_KEY or not CX:
            raise HTTPException(
                status_code=500, 
                detail="Google API configuration missing"
            )

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": request.query,
            "cx": CX,
            "searchType": "image",
            "key": API_KEY,
            "num": 1
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Error fetching image from Google API"
                )

            data = response.json()
            image_url = data.get("items", [{}])[0].get("link") if data.get("items") else None

            if not image_url:
                raise HTTPException(
                    status_code=404,
                    detail="No image found for the query"
                )

            return {
                "query": request.query,
                "imageUrl": image_url
            }

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Update the generate_menu endpoint
@app.post("/api/generate-menu/{uid}")
async def generate_menu(uid: str):
    """Generate menu items based on user preferences from Firebase"""
    try:
        print(f"\n=== Starting Menu Generation for User {uid} ===")

        # Fetch user preferences from Firebase
        user_ref = db_firebase.collection('users').document(uid)
        user_prefs = user_ref.collection('AllAboutUser').document('preferences').get()

        if not user_prefs.exists:
            raise HTTPException(status_code=404, detail="User preferences not found")

        preferences = user_prefs.to_dict()
        print("\nUser Preferences:", json.dumps(preferences, indent=2))

        # Create OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("\nCreating recipes with OpenAI...")

        # Generate recipes
        completion = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional chef creating unique recipes based on user preferences. Return only valid JSON arrays. Include realistic pricing for each recipe based on ingredient costs and complexity."
                },
                {
                    "role": "user",
                    "content": f"""
                    Create 10 unique recipes for a user with these preferences:
                    - Cuisine Preferences: {preferences.get('cuisinePreferences', [])}
                    - Dietary Preferences: {preferences.get('dietaryPreferences', [])}
                    - Food Allergies: {preferences.get('foodAllergies', [])}
                    - Location: {preferences.get('location', 'Unknown')}

                    Return ONLY a JSON array with this exact structure:
                    [
                        {{
                            "name": "Recipe Name",
                            "description": "Brief but appealing description",
                            "ingredients": [
                                {{"id": "1", "name": "Ingredient Name", "quantity": "amount", "price": price_in_dollars}}
                            ],
                            "dietaryInfo": {{
                                "isVegetarian": boolean,
                                "isVegan": boolean,
                                "isGlutenFree": boolean,
                                "isDairyFree": boolean
                            }},
                            "totalPrice": total_cost_in_dollars,
                            "priceCategory": "budget"|"moderate"|"premium"
                        }}
                    ]

                    Requirements:
                    1. Generate exactly 10 unique recipes
                    2. Each recipe must be different
                    3. Respect all dietary restrictions and allergies
                    4. Match preferred cuisines when possible
                    5. Include precise quantities
                    6. Include realistic prices for each ingredient (e.g., $1.99, $3.50)
                    7. Calculate a total price that reflects the sum of all ingredient costs
                    8. Assign a price category: "budget" (<$15), "moderate" ($15-30), or "premium" (>$30)
                    9. Ensure all recipes are practical and well-balanced
                    10. Return only the JSON array, no additional text
                    """
                }
            ],
            max_tokens=4000,
            temperature=0.7
        )

        # Get and clean the response
        response = completion.choices[0].message.content.strip()
        print("\nReceived response from OpenAI")
        print("Response length:", len(response))

        try:
            # Clean the response
            if '```json' in response:
                response = response.split('```json')[-1].split('```')[0].strip()
            
            print("\nAttempting to parse JSON...")
            recipes_data = json.loads(response)
            
            if not isinstance(recipes_data, list):
                raise ValueError("Response is not a list of recipes")
            
            print(f"\nValidated {len(recipes_data)} recipes")

            # Process all recipes
            menu_items = []
            for recipe in recipes_data:
                try:
                    # Generate image URL
                    print(f"\nGenerating image for: {recipe['name']}")
                    image_result = await search_image(
                        ImageSearchRequest(query=f"{recipe['name']} food dish")
                    )
                    
                    # Create menu item with UUID
                    recipe_id = str(uuid.uuid4())
                    menu_item = {
                        "id": recipe_id,
                        "name": recipe["name"],
                        "description": recipe["description"],
                        "image": image_result.get("imageUrl"),
                        "ingredients": recipe["ingredients"],
                        "dietaryInfo": recipe["dietaryInfo"],
                        "totalPrice": recipe.get("totalPrice", 0.0),
                        "priceCategory": recipe.get("priceCategory", "moderate"),
                        "rating": 4.5
                    }

                    # Store in Firebase
                    menu_ref = user_ref.collection('menu_items').document(recipe_id)
                    menu_ref.set(menu_item)
                    
                    menu_items.append(menu_item)
                    print(f"Created menu item: {menu_item['name']} - ${menu_item['totalPrice']:.2f} ({menu_item['priceCategory']})")

                except Exception as e:
                    print(f"Error processing recipe {recipe.get('name', 'unknown')}: {str(e)}")
                    continue

            print(f"\n=== Generated {len(menu_items)} menu items successfully ===")
            return {
                "message": f"Generated {len(menu_items)} menu items",
                "menu_items": menu_items
            }

        except json.JSONDecodeError as e:
            print(f"\nJSON Parse Error: {str(e)}")
            print("Raw response:", response)
            raise HTTPException(status_code=500, detail=f"Failed to parse recipe data: {str(e)}")

    except Exception as e:
        print(f"\n=== Error generating menu ===")
        print(f"Type: {type(e)}")
        print(f"Details: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_tb(e.__traceback__)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/menu-items/{uid}/{item_id}")
async def delete_menu_item(uid: str, item_id: str):
    """Delete a specific menu item"""
    try:
        menu_ref = db_firebase.collection('users').document(uid).collection('menu_items').document(item_id)
        if not (await menu_ref.get()).exists:
            raise HTTPException(status_code=404, detail="Menu item not found")
        
        await menu_ref.delete()
        return {"message": "Menu item deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete menu item: {str(e)}")

@app.get("/debug/firebase-users")
async def debug_firebase_users():
    """Debug endpoint to list first 5 users in Firebase"""
    try:
        print("\n=== Starting Firebase Users Debug ===")
        
        # Get users collection reference
        users_ref = db_firebase.collection('users')
        
        print("\nAttempting to list all users...")
        users = users_ref.limit(5).stream()  # Limit to first 5 users
        
        debug_info = {
            "user_count": 0,
            "users": []
        }
        
        print("\nProcessing users:")
        for user in users:
            user_data = {
                "user_id": user.id,
                "exists": user.exists,
                "path": user.reference.path,
            }
            
            # Check for preferences
            prefs_ref = user.reference.collection('AllAboutUser').document('preferences')
            prefs_doc = prefs_ref.get()
            user_data["has_preferences"] = prefs_doc.exists
            
            if prefs_doc.exists:
                user_data["preferences"] = prefs_doc.to_dict()
            
            print(f"\nUser {user.id}:")
            print(f"- Path: {user.reference.path}")
            print(f"- Has preferences: {prefs_doc.exists}")
            
            # Check for menu items
            menu_items = user.reference.collection('menu_items').stream()
            menu_items_list = [item.to_dict() for item in menu_items]
            user_data["menu_items_count"] = len(menu_items_list)
            
            print(f"- Menu items count: {user_data['menu_items_count']}")
            
            debug_info["users"].append(user_data)
            debug_info["user_count"] += 1
        
        print(f"\nTotal users found: {debug_info['user_count']}")
        
        # Add collection names debug
        collections = db_firebase.collections()
        debug_info["available_collections"] = [col.id for col in collections]
        print(f"\nAvailable collections: {debug_info['available_collections']}")
        
        return debug_info

    except Exception as e:
        error_msg = f"Firebase debug error: {str(e)}"
        print(f"\n=== ERROR ===\n{error_msg}")
        print(f"Error type: {type(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            print("Traceback:")
            traceback.print_tb(e.__traceback__)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/debug/create-test-user")
async def create_test_user():
    """Create a test user in Firebase to verify write access"""
    try:
        print("\n=== Creating Test User ===")
        
        # Generate test user ID
        test_uid = f"test_user_{uuid.uuid4().hex[:8]}"
        
        # Create user document
        user_ref = db_firebase.collection('users').document(test_uid)
        
        # Create test user data
        user_data = {
            "created_at": datetime.datetime.now(),
            "test_field": True
        }
        
        # Create preferences data
        preferences_data = {
            "cuisinePreferences": ["Italian", "Japanese"],
            "foodAllergies": ["Nuts"],
            "dietaryPreferences": ["Vegetarian"],
            "location": "San Diego, CA"
        }
        
        print(f"\nAttempting to create user: {test_uid}")
        print("Writing user document...")
        user_ref.set(user_data)
        
        print("Writing preferences document...")
        prefs_ref = user_ref.collection('AllAboutUser').document('preferences')
        prefs_ref.set(preferences_data)
        
        return {
            "message": "Test user created successfully",
            "uid": test_uid,
            "user_data": user_data,
            "preferences": preferences_data
        }

    except Exception as e:
        print(f"\n=== ERROR Creating Test User ===")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create test user: {str(e)}")

@app.post("/api/generate-categories/{uid}")
async def generate_categories(uid: str):
    """Generate categorized product listings based on user preferences"""
    try:
        print(f"\n=== Starting Category Generation for User {uid} ===")

        # Fetch user preferences from Firebase
        user_ref = db_firebase.collection('users').document(uid)
        user_prefs = user_ref.collection('AllAboutUser').document('preferences').get()

        if not user_prefs.exists:
            raise HTTPException(status_code=404, detail="User preferences not found")

        preferences = user_prefs.to_dict()
        print("\nUser Preferences:", json.dumps(preferences, indent=2))

        # Create OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Generate categories and products
        print("\nGenerating catalog with OpenAI...")
        completion = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a retail categorization expert. Generate a valid JSON catalog of grocery categories and products with detailed pricing information and price tiers."
                },
                {
                    "role": "user",
                    "content": f"""
                    Create a grocery store catalog for a user with these preferences:
                    - Cuisine Preferences: {preferences.get('cuisinePreferences', [])}
                    - Dietary Preferences: {preferences.get('dietaryPreferences', [])}
                    - Food Allergies: {preferences.get('foodAllergies', [])}
                    - Location: {preferences.get('location', 'Unknown')}

                    Return ONLY a JSON object with this exact structure:
                    {{
                        "categories": [
                            {{
                                "id": "cat1",
                                "name": "Category Name",
                                "subcategories": ["Subcategory 1", "Subcategory 2"],
                                "priceRange": {{"min": 1.99, "max": 15.99}}
                            }}
                        ],
                        "products": [
                            {{
                                "id": "prod1",
                                "name": "Product Name",
                                "price": 9.99,
                                "salePrice": null,
                                "pricePerUnit": "2.49/lb",
                                "priceCategory": "value"|"standard"|"premium",
                                "description": "Product description",
                                "category": "Category Name",
                                "subcategory": "Subcategory 1",
                                "nutritionalValue": "High in protein, low in fat",
                                "origin": "Country or region of origin",
                                "organic": true|false
                            }}
                        ]
                    }}

                    Requirements:
                    1. Generate at least 7 main categories (e.g., Produce, Dairy, Bakery)
                    2. Each category should have 3-5 subcategories
                    3. Generate 3-5 products per subcategory
                    4. Ensure all products respect dietary restrictions
                    5. Use realistic prices based on current market values
                    6. Include price categories ("value", "standard", "premium")
                    7. Add sale prices to approximately 30% of products
                    8. Include organic options for relevant categories
                    9. Add nutritional value highlights
                    10. Include origin information for products
                    11. Ensure all IDs are unique strings
                    12. Return only the JSON object, no additional text
                    """
                }
            ],
            max_tokens=4000,
            temperature=0.7
        )

        # Get and clean the response
        response = completion.choices[0].message.content.strip()
        print("\nReceived response from OpenAI")
        print("Response length:", len(response))

        try:
            # Clean the response
            if '```json' in response:
                response = response.split('```json')[-1].split('```')[0].strip()
            
            print("\nAttempting to parse JSON...")
            catalog_data = json.loads(response)
            
            # Validate the structure
            if not isinstance(catalog_data, dict):
                raise ValueError("Response is not a dictionary")
            if 'categories' not in catalog_data or 'products' not in catalog_data:
                raise ValueError("Missing required fields: categories or products")
            
            print(f"\nValidated {len(catalog_data['categories'])} categories")
            print(f"Validated {len(catalog_data['products'])} products")

            # Generate images for categories
            print("\nGenerating images for categories...")
            for category in catalog_data['categories']:
                try:
                    image_result = await search_image(
                        ImageSearchRequest(query=f"{category['name']} grocery category")
                    )
                    category['image'] = image_result.get('imageUrl')
                    print(f"Added image for category: {category['name']}")
                except Exception as img_error:
                    print(f"Error getting image for category {category['name']}: {str(img_error)}")
                    category['image'] = None

            # Generate images for products
            print("\nGenerating images for products...")
            for product in catalog_data['products']:
                try:
                    image_result = await search_image(
                        ImageSearchRequest(query=f"{product['name']} food product")
                    )
                    product['image'] = image_result.get('imageUrl')
                    print(f"Added image for product: {product['name']} - ${product['price']:.2f} ({product['priceCategory']})")
                except Exception as img_error:
                    print(f"Error getting image for product {product['name']}: {str(img_error)}")
                    product['image'] = None

            # Store in Firebase
            print("\nStoring catalog in Firebase...")
            catalog_ref = user_ref.collection('catalog')
            
            # Store categories
            categories_doc = catalog_ref.document('categories')
            categories_doc.set({"items": catalog_data['categories']})
            print("Stored categories in Firebase")
            
            # Store products
            products_doc = catalog_ref.document('products')
            products_doc.set({"items": catalog_data['products']})
            print("Stored products in Firebase")

            print("\n=== Category Generation Complete ===")
            return {
                "message": "Catalog generated successfully",
                "categories": catalog_data['categories'],
                "products": catalog_data['products']
            }

        except json.JSONDecodeError as e:
            print(f"\nJSON Parse Error: {str(e)}")
            print("Raw response:", response)
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse catalog data: {str(e)}"
            )
        except ValueError as e:
            print(f"\nValidation Error: {str(e)}")
            print("Parsed data:", catalog_data)
            raise HTTPException(
                status_code=500, 
                detail=f"Invalid catalog data: {str(e)}"
            )

    except Exception as e:
        print(f"\n=== Error generating catalog ===")
        print(f"Type: {type(e)}")
        print(f"Details: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_tb(e.__traceback__)
        raise HTTPException(status_code=500, detail=str(e))

# Add endpoint to retrieve catalog
@app.get("/api/categories/{uid}")
async def get_categories(uid: str):
    """Retrieve user's categorized product catalog"""
    try:
        catalog_ref = db_firebase.collection('users').document(uid).collection('catalog')
        
        # Get categories
        categories_doc = catalog_ref.document('categories').get()
        products_doc = catalog_ref.document('products').get()
        
        if not categories_doc.exists or not products_doc.exists:
            raise HTTPException(status_code=404, detail="Catalog not found")
            
        return {
            "categories": categories_doc.to_dict()['items'],
            "products": products_doc.to_dict()['items']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch catalog: {str(e)}")

# Add this after creating the FastAPI app
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/api/menu-items/{uid}")
async def get_menu_items(uid: str):
    """Retrieve user's generated menu items"""
    try:
        print(f"\n=== Retrieving Menu Items for User {uid} ===")
        
        user_ref = db_firebase.collection('users').document(uid)
        menu_items_collection = user_ref.collection('menu_items')
        
        # Get all menu items for the user
        menu_items_docs = menu_items_collection.stream()
        
        menu_items = []
        for doc in menu_items_docs:
            item_data = doc.to_dict()
            menu_items.append(item_data)
        
        if not menu_items:
            print(f"No menu items found for user {uid}")
            return {
                "message": "No menu items found",
                "menu_items": []
            }
            
        print(f"Retrieved {len(menu_items)} menu items for user {uid}")
        
        return {
            "message": f"Retrieved {len(menu_items)} menu items",
            "menu_items": menu_items
        }

    except Exception as e:
        print(f"\n=== Error retrieving menu items ===")
        print(f"Type: {type(e)}")
        print(f"Details: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_tb(e.__traceback__)
        raise HTTPException(status_code=500, detail=f"Failed to fetch menu items: {str(e)}")