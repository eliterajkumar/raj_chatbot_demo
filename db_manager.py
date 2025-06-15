import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import logging

logger = logging.getLogger(__name__)

# --- Configuration for MongoDB Connection ---
# It's best practice to use environment variables for sensitive information
# like database connection strings.
# For local development, you can set these directly or use a .env file.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "fynorra_chatbot_db")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "chat_history")

# --- MongoDB Client Initialization ---
# Using a global client for efficiency across requests.
# It's important to initialize this client when your application starts.
client = None
db = None
collection = None

def initialize_db():
    """Initializes the MongoDB client and collection."""
    global client, db, collection
    if client is not None:
        logger.info("MongoDB client already initialized.")
        return

    try:
        logger.info(f"Attempting to connect to MongoDB at {MONGO_URI}...")
        client = MongoClient(MONGO_URI)
        # The ping command is cheap and does not require auth.
        client.admin.command('ping')
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]
        logger.info("Successfully connected to MongoDB!")
    except ConnectionFailure as e:
        logger.error(f"MongoDB Connection failed: {e}. Ensure MongoDB is running and accessible.")
        client = None
        db = None
        collection = None
        raise ConnectionFailure(f"Could not connect to MongoDB: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during MongoDB initialization: {e}")
        client = None
        db = None
        collection = None
        raise e

def close_db():
    """Closes the MongoDB connection."""
    global client, db, collection
    if client:
        client.close()
        client = None
        db = None
        collection = None
        logger.info("MongoDB connection closed.")

def get_chat_history(chat_id: str) -> list[tuple[str, str]]:
    """
    Retrieves chat history for a given chat_id from MongoDB.
    Returns a list of (user_message, ai_response) tuples.
    """
    if collection is None:
        logger.error("MongoDB collection not initialized. Cannot retrieve chat history.")
        return []

    try:
        # Find the document for the specific chat_id
        doc = collection.find_one({"_id": chat_id})
        if doc and "history" in doc:
            # Convert list of dicts back to list of tuples
            history = [(item["question"], item["answer"]) for item in doc["history"]]
            logger.debug(f"Retrieved chat history for {chat_id}: {history}")
            return history
        logger.debug(f"No chat history found for {chat_id}")
        return []
    except OperationFailure as e:
        logger.error(f"MongoDB OperationFailure during get_chat_history for {chat_id}: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during get_chat_history for {chat_id}: {e}")
        return []

def save_chat_history(chat_id: str, history: list[tuple[str, str]]):
    """
    Saves or updates chat history for a given chat_id in MongoDB.
    'history' should be a list of (user_message, ai_response) tuples.
    """
    if collection is None:
        logger.error("MongoDB collection not initialized. Cannot save chat history.")
        return

    try:
        # MongoDB doesn't directly store tuples in documents.
        # Convert list of tuples to list of dictionaries for storage.
        history_for_db = [{"question": q, "answer": a} for q, a in history]

        # Use upsert=True to insert if _id doesn't exist, or update if it does.
        collection.update_one(
            {"_id": chat_id},
            {"$set": {"history": history_for_db, "last_updated": datetime.now()}},
            upsert=True
        )
        logger.debug(f"Saved chat history for {chat_id}: {history}")
    except OperationFailure as e:
        logger.error(f"MongoDB OperationFailure during save_chat_history for {chat_id}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during save_chat_history for {chat_id}: {e}")

# Optional: Ensure datetime is imported for 'last_updated'
from datetime import datetime

# It's good practice to call initialize_db when your app starts
# This will be done in main.py, but keep it in mind for testing this module.
