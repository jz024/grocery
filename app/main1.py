import os
import firebase_admin
from firebase_admin import credentials, firestore, auth
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import datetime
import json
import requests
from fastapi.middleware.cors import CORSMiddleware

# 🔹 Set the correct path for Firebase credentials
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of main1.py
FIREBASE_KEY_PATH = os.path.join(BASE_DIR, "../venv/lib/firebase.js")  # Adjust path

# 🔹 Initialize Firebase Admin SDK only once
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred)

# 🔹 Firestore Database
db = firestore.client()
onboarding_collection = db.collection("users")
chat_collection = db.collection("chatHistory")

# 🔹 FastAPI App
app = FastAPI()

# 🔹 CORS Middleware (Allow Frontend to Connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Security for Firebase Authentication
security = HTTPBearer()

def verify_firebase_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Middleware function to verify Firebase authentication token.
    """
    token = credentials.credentials
    try:
        decoded_token = auth.verify_id_token(token)  # Verify Firebase ID token
        uid = decoded_token.get("uid")  # Extract user ID
        return uid
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")

# 🔹 User Preferences Schema
class UserPreferences(BaseModel):
    dietaryPreferences: Optional[List[str]] = None
    foodAllergies: Optional[List[str]] = None
    cuisinePreferences: Optional[List[str]] = None
    location: Optional[Dict[str, float]] = None  # latitude & longitude
    preferences: Optional[Dict[str, Any]] = None 

@app.post("/onboarding")
def update_user_preferences(data: UserPreferences, uid: str = Depends(verify_firebase_token)):
    try:
        user_ref = onboarding_collection.document(uid)

        update_fields = {
            "dietaryPreferences": data.dietaryPreferences,
            "foodAllergies": data.foodAllergies,
            "cuisinePreferences": data.cuisinePreferences,
            "location": data.location,
            "preferences": data.preferences,
            "onboardingCompleted": True
        }

        user_ref.set(update_fields, merge=True)

        return {"message": "User preferences updated successfully", "updated_fields": update_fields}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/onboarding/{uid}")
def get_user_preferences(uid: str, current_uid: str = Depends(verify_firebase_token)):
    user_data = onboarding_collection.document(uid).get()
    if not user_data.exists:
        raise HTTPException(status_code=404, detail="User preferences not found")
    return user_data.to_dict()

@app.post("/chat")
def chat_with_ai(message: str, uid: str = Depends(verify_firebase_token)):
    try:
        # Fetch user preferences
        user_data = onboarding_collection.document(uid).get()
        user_prefs = user_data.to_dict() if user_data.exists else {}

        # Fetch recent chat history (last 5 messages)
        chat_history = chat_collection.where("uid", "==", uid).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
        chat_history = [{"user_message": chat.to_dict().get("user_message"), "ai_response": chat.to_dict().get("ai_response")} for chat in chat_history]
        
        chat_history.reverse()

        messages = [
            {"role": "system", "content": "You are a helpful grocery assistant. Provide specific and practical advice about groceries, recipes, and shopping while considering the user's preferences and restrictions."}
        ]

        for chat in chat_history:
            messages.extend([
                {"role": "user", "content": chat["user_message"]},
                {"role": "assistant", "content": chat["ai_response"]}
            ])

        current_context = f"""
        User Preferences:
        {json.dumps(user_prefs, indent=2)}
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

        chat_collection.add({
            "uid": uid,
            "type": "chat",
            "user_message": message,
            "ai_response": ai_response,
            "timestamp": firestore.SERVER_TIMESTAMP
        })

        return {"ai_response": ai_response}

    except Exception as e:
        print(f"Chat error: {str(e)}")
        return {"ai_response": f"I encountered an error: {str(e)}"}
