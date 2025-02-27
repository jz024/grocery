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

# Initialize Firebase Admin
cred = credentials.Certificate(os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH"))
firebase_admin.initialize_app(cred)
db_firebase = firestore.client()

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
        # Fetch user preferences from Firebase
        user_ref = db_firebase.collection('users').document(uid).collection('AllAboutUser').document('preferences')
        user_doc = user_ref.get()
        
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
        history = list(chat_collection.find(
            {
                "uid": uid,
                "type": {"$in": ["chat", "shopping_list"]}  # Include both types
            },
            {"_id": 0, "type": 1, "user_message": 1, "ai_response": 1, "list_data": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(5))  # Increased limit to include more context
        
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
        completion = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )

        ai_response = completion.choices[0].message.content.strip()

        # Save chat history
        chat_collection.insert_one({
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
                "type": "chat"
            },
            {"_id": 0, "user_message": 1, "ai_response": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(10))

        if not chat_history:
            return {"locations": [], "items": []}

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

        # print("\n=== Current Preferences ===")
        # print(json.dumps(current_preferences, indent=2))

        # Analyze chat history for new preferences
        client = OpenAI()
        chat_text = "\n".join([
            f"User: {chat.get('user_message', '')}\nAssistant: {chat.get('ai_response', '')}"
            for chat in reversed(chat_history)
        ])

        # print("\n=== Analyzing Chat History ===")
        # print(chat_text)

        preference_prompt = f"""
        Analyze this conversation and identify any new or updated user preferences.
        Current preferences: {current_preferences}

        Conversation:
        {chat_text}

        Return a valid JSON object with only new or changed preferences:
        {{
            "cuisinePreferences": ["new cuisine 1", "new cuisine 2"],
            "foodAllergies": ["new allergy 1"],
            "dietaryPreferences": ["new diet 1"]
        }}
        Only include preferences that are clearly stated or implied in the conversation.
        Return empty arrays if no new preferences are found.
        """

        preference_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a preference analyzer that returns only JSON."},
                {"role": "user", "content": preference_prompt}
            ],
            max_tokens= 1000,
            temperature=0.3
        )

        try:
            # Clean the response by removing markdown code block markers if present
            raw_response = preference_completion.choices[0].message.content.strip()
            cleaned_response = raw_response.replace('```json', '').replace('```', '').strip()
            
            new_preferences = json.loads(cleaned_response)
            
            print("\n=== New Preferences Detected ===")
            print(json.dumps(new_preferences, indent=2))
            
            # Merge new preferences with existing ones
            for key in ['cuisinePreferences', 'foodAllergies', 'dietaryPreferences']:
                if key in new_preferences and new_preferences[key]:
                    current_set = set(current_preferences.get(key, []))
                    new_set = set(new_preferences[key])
                    current_preferences[key] = list(current_set.union(new_set))

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
                    
                    # Check if all new preferences were saved
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
                # Continue with shopping list generation even if Firebase update fails

        except json.JSONDecodeError as e:
            print("\n=== Error Parsing New Preferences ===")
            print(f"Error: {str(e)}")
            print("Raw response:", preference_completion.choices[0].message.content.strip())

        # Now continue with shopping list generation using updated preferences
        formatted_chats = "\n".join([
            f"User: {chat['user_message']}\nAssistant: {chat['ai_response']}"
            for chat in reversed(chat_history)
        ])

        # Construct prompt for shopping list generation
        prompt = f"""
        Based on the following conversation history and user preferences, create a shopping list with store locations and items.
        The list should include all of the items that the user needs to purchase to cook the meal reccomended to them. 
        This should not include any items that were mentioned for other options that the user did not choose.
        Only include items that were specifically discussed or recommended in the conversation.
        If prices were mentioned, include them; otherwise, estimate reasonable prices.
        As for locations, please use your best judgement based on the user's location.
        the output locations should be actual store names, for example "target, walmart, Ranch99, tarder joes, etc"
        User Preferences:
        {json.dumps(current_preferences, indent=2)}  

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
            
            valid_items = []
            for item in shopping_list["items"]:
                if isinstance(item, dict) and "name" in item and "price" in item:
                    if isinstance(item["name"], str) and isinstance(item["price"], (int, float)):
                        valid_items.append(item)
            
            shopping_list["items"] = valid_items

            # Save the generated shopping list
            chat_collection.insert_one({
                "uid": uid,
                "type": "shopping_list",
                "list_data": shopping_list,
                "timestamp": datetime.datetime.now()
            })

            # # Delete chat history for this user
            # delete_result = chat_collection.delete_many({
            #     "uid": uid,
            #     "type": "chat"  # Only delete chat messages, keep shopping lists
            # })
            
            # print(f"\n=== Deleted {delete_result.deleted_count} chat messages for user {uid} ===")

            return shopping_list

        except json.JSONDecodeError:
            return {"locations": [], "items": []}

    except Exception as e:
        print(f"Shopping list generation error: {str(e)}")
        return {"locations": [], "items": []}

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
        chat_collection.insert_one({
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

# Add this after creating the FastAPI app
app.mount("/static", StaticFiles(directory="app/static"), name="static")