from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from recommender_net import RecommenderNet
from tensorflow.keras.models import load_model
import random

# Initialize FastAPI app
app = FastAPI()

# Load trained model
model_path = "./model/collaborative_filtering.keras"
try:
    recommender_model = load_model(model_path)
    #recommender_model = tf.keras.models.load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Load encoders
encoder_path = "./model/encoders.json"
try:
    with open(encoder_path, "r") as f:
        encoders = json.load(f)
except FileNotFoundError:
    raise RuntimeError("Encoders file not found. Ensure 'encoders.json' is in the specified path.")

encode_user_id1 = encoders['encode_user_id1']
encoded_user_id2 = {v: k for k, v in encode_user_id1.items()}

encode_title1 = encoders['encode_title1']
encoded_title2 = {v: k for k, v in encode_title1.items()}

# Load data
data_path = "./data/book-ratings.csv"
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    raise RuntimeError("Data file not found. Ensure 'book-ratings.csv' is in the specified path.")

class RecommendationResponse(BaseModel):
    user_id: str
    recommended_books: list

@app.get("/recommend", response_model=RecommendationResponse)
def recommend_books():
    """
    Automatically generate book recommendations for a randomly chosen user.
    """
    # Randomly select a user
    user_id = random.choice(df['User-ID'].unique())

    # Check if the user exists in the encoders
    if user_id not in encode_user_id1:
        raise HTTPException(status_code=404, detail="User ID not found")

    # Encode the user
    user_encoded = encode_user_id1[user_id]

    # Identify unrated books
    all_books = list(encode_title1.keys())
    rated_books = df[df['User-ID'] == user_id]['Book-Title'].tolist()
    unrated_books = [book for book in all_books if book not in rated_books]

    if not unrated_books:
        return RecommendationResponse(user_id=user_id, recommended_books=[])

    # Encode unrated books
    unrated_books_encoded = [[encode_title1[book]] for book in unrated_books]

    # Generate predictions
    user_books_array = np.hstack(([[user_encoded]] * len(unrated_books_encoded), unrated_books_encoded))
    predicted_ratings = recommender_model.predict(user_books_array).flatten()

    # Get top 10 recommendations
    top_indices = predicted_ratings.argsort()[-10:][::-1]
    recommendations = [encoded_title2[unrated_books_encoded[idx][0]] for idx in top_indices]

    return RecommendationResponse(user_id=user_id, recommended_books=recommendations)
