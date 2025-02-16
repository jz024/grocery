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

load_dotenv()

# Load environment variables
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

# MongoDB Connection
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client.grocerydb
onboarding_collection = db.onboardingData
chat_collection = db.chatHistory

# Updated User Preferences Schema
class UserPreferences(BaseModel):
    uid: str
    cuisinePreferences: Optional[List[str]] = None
    foodAllergies: Optional[List[str]] = None
    dietaryPreferences: Optional[List[str]] = None
    preferences: Optional[Dict[str, Any]] = None 

@app.post("/onboarding")
def update_user_preferences(data: UserPreferences):
    try:
        if not data.uid:
            raise HTTPException(status_code=400, detail="User ID (uid) is required")

        existing_user = onboarding_collection.find_one({"uid": data.uid})

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

@app.post("/chat")
def chat_with_ai(uid: str, message: str):
    try:
        # Fetch user preferences
        user_data = {
                "cuisinePreferences": ["Italian", "Mexican", "Asian"],
                "foodAllergies": ["Peanuts", "Shellfish"],
                "dietaryPreferences": ["Vegetarian", "Low-carb"],
                "location": "San Francisco, CA",
                "preferences": {
                    "organic": True,
                    "localProduce": True,
                    "budgetLevel": "medium",
                    "cookingTime": "30-60min"
                }
        }

        # Fetch recent chat history (last 5 messages)
        chat_history = list(chat_collection.find(
            {
                "uid": uid,
                "type": "chat"  # Only get chat messages, not shopping lists
            }, 
            {"_id": 0, "user_message": 1, "ai_response": 1}
        ).sort("timestamp", -1).limit(5))
        
        # Reverse to get chronological order
        chat_history.reverse()

        # Construct messages array with chat history
        messages = [
            {"role": "system", "content": "You are a helpful grocery assistant. Provide specific and practical advice about groceries, recipes, and shopping while considering the user's preferences and restrictions."}
        ]

        # Add chat history to messages
        for chat in chat_history:
            if "user_message" in chat and "ai_response" in chat:  # Check if fields exist
                messages.extend([
                    {"role": "user", "content": chat["user_message"]},
                    {"role": "assistant", "content": chat["ai_response"]}
                ])

        # Add current user preferences and message
        current_context = f"""
        User Preferences:
        - Cuisine Preferences: {user_data.get("cuisinePreferences", [])}
        - Food Allergies: {user_data.get("foodAllergies", [])}
        - Dietary Preferences: {user_data.get("dietaryPreferences", [])}
        - Preferences: {user_data.get("preferences", {})}
        - Location: {user_data.get("location", "San Francisco, CA")}
        Current message: {message}
        """
        messages.append({"role": "user", "content": current_context})

        client = OpenAI()

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )

        ai_response = completion.choices[0].message.content.strip()

        # Save chat history with type field
        chat_collection.insert_one({
            "uid": uid, 
            "type": "chat",  # Add type field to distinguish from shopping lists
            "user_message": message, 
            "ai_response": ai_response,
            "timestamp": datetime.datetime.now()
        })

        return {"ai_response": ai_response}

    except Exception as e:
        print(f"Chat error: {str(e)}")  # Log the full error
        return {"ai_response": f"I apologize, but I encountered an error: {str(e)}"}

@app.get("/directions")
def get_directions(locations: List[str]):
    try:
        if len(locations) < 2:
            raise HTTPException(status_code=400, detail="At least two locations are required")

        base_url = "https://maps.googleapis.com/maps/api/directions/json"
        waypoints = "|".join(locations[1:-1])

        params = {
            "origin": locations[0],
            "destination": locations[-1],
            "waypoints": waypoints,
            "key": GOOGLE_MAPS_API_KEY,
        }

        response = requests.get(base_url, params=params)
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
def generate_shopping_list(uid: str):
    try:
        # Fetch recent chat history
        chat_history = list(chat_collection.find(
            {
                "uid": uid,
                "type": "chat"  # Only get chat messages, not shopping lists
            },
            {"_id": 0, "user_message": 1, "ai_response": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(10))

        if not chat_history:
            # Return empty shopping list structure instead of raising an error
            return {
                "locations": [],
                "items": []
            }

        # Format chat history for the LLM
        formatted_chats = "\n".join([
            f"User: {chat['user_message']}\nAssistant: {chat['ai_response']}"
            for chat in reversed(chat_history)
        ])

        # Construct prompt for shopping list generation
        prompt = f"""
        Based on the following conversation history, create a shopping list with store locations and items.
        Only include items that were specifically discussed or recommended in the conversation.
        If prices were mentioned, include them; otherwise, estimate reasonable prices.
        As for locations, please use your best judgement based on the user's location.  

        Conversation History:
        {formatted_chats}

        Return a valid JSON object with exactly this structure:
        {{
            "locations": ["store1", "store2"],
            "items": [
                {{"name": "item1", "price": 0.00, "location": "store1"}},
                {{"name": "item2", "price": 0.00, "location": "store2"}}
            ]
        }}

        Important:
        1. Ensure the response is valid JSON
        2. Use only double quotes for strings
        3. Use numbers without quotes for prices
        4. Include at least one store and one item
        5. If no specific items were discussed, return an empty list
        6. Do not include any comments or explanations, only the JSON object
        """

        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a JSON-generating assistant that creates shopping lists. Always return valid JSON without any additional text or comments."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )

        # Get the response and clean it
        response_text = completion.choices[0].message.content.strip()
        
        # Try to parse the JSON
        try:
            shopping_list = json.loads(response_text)
            
            # Validate the structure
            if not isinstance(shopping_list, dict):
                return {"locations": [], "items": []}
            
            # Ensure required fields exist
            shopping_list.setdefault("locations", [])
            shopping_list.setdefault("items", [])
            
            # Validate types
            if not isinstance(shopping_list["locations"], list):
                shopping_list["locations"] = []
            if not isinstance(shopping_list["items"], list):
                shopping_list["items"] = []
            
            # Validate each item
            valid_items = []
            for item in shopping_list["items"]:
                try:
                    if isinstance(item, dict) and "name" in item and "price" in item:
                        if isinstance(item["name"], str) and isinstance(item["price"], (int, float)):
                            valid_items.append(item)
                except:
                    continue
            
            shopping_list["items"] = valid_items

            # Save the generated shopping list to the database
            chat_collection.insert_one({
                "uid": uid,
                "type": "shopping_list",
                "list_data": shopping_list,
                "timestamp": datetime.datetime.now()
            })

            # Delete chat history for this user
            chat_collection.delete_many({
                "uid": uid,
                "type": "chat"  # Only delete chat messages, keep shopping lists
            })

            return shopping_list

        except json.JSONDecodeError as json_error:
            print(f"JSON parsing error: {str(json_error)}")
            print(f"Raw response: {response_text}")
            # Return empty structure instead of error
            return {"locations": [], "items": []}

        except ValueError as val_error:
            print(f"Validation error: {str(val_error)}")
            # Return empty structure instead of error
            return {"locations": [], "items": []}

    except Exception as e:
        print(f"Shopping list generation error: {str(e)}")
        # Return empty structure instead of error
        return {"locations": [], "items": []}

# Add this after creating the FastAPI app
app.mount("/static", StaticFiles(directory="app/static"), name="static")
