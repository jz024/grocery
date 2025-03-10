
## Requirements

- Python 3.8+
- MongoDB
- Google Maps Places API Key
- OpenAI API Key
- Perplexity API Key
- Firebase

---

## Setting Up API Keys

### 1. **Google Maps Places API Key Setup**

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a **new project** (if you don’t already have one).
3. Navigate to **APIs & Services > Library** and search for:
   - **Places API**
   - **Geocoding API** (optional, if needed for geolocation)
4. Enable the APIs.
5. Navigate to **APIs & Services > Credentials**, and create a new API Key.
6. Restrict the API key (recommended for security) to specific APIs and HTTP referrers or IP addresses.
7. Copy the API Key and add it to your `.env` file:
   ```plaintext
   GOOGLE_MAPS_API_KEY=your_google_maps_api_key
   ```

### 2. **OpenAI API Key Setup**

1. Go to the [OpenAI API page](https://platform.openai.com/signup).
2. Log in or sign up if you don’t already have an account.
3. Navigate to **API Keys** under your account settings.
4. Create a new API key and copy it.
5. Add it to your `.env` file:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

### 3. **Perplexity API Key Setup**

1. Go to the [Perplexity API page](https://www.perplexity.ai/).  
2. Log in or sign up if you don’t already have an account.  
3. Navigate to **API Keys** under your account settings.  
4. Create a new API key and copy it.  
5. Add it to your `.env` file:  
   ```plaintext
   PERPLEXITY_API_KEY=your_perplexity_api_key
   ```


---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/grocery-recommender-api.git
   cd grocery-recommender-api
   ```

2. **Create a Conda Environment:**

   ```bash
   conda create -n grocery_env python=3.8 -y
   conda activate grocery_env
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   Create a `.env` file in the root directory and add:

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_MAPS_API_KEY=your_google_maps_api_key
   MONGODB_URI=your_mongodb_connection_uri
   PERPLEXITY_API_KEY=your_perplexity_api_key
   FIREBASE_SERVICE_ACCOUNT_PATH=you_firebase_credential_file
   ```

---

## ▶️ Running the Application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

---
