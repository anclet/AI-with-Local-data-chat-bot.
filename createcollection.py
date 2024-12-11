import os
from qdrant_client import QdrantClient, models as rest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Initialize Qdrant client
#client = QdrantClient(url="http://localhost", port=6333)

#client = QdrantClient(url="http://host.docker.internal", port=6333)
client = QdrantClient(url="http://localhost", port=6333)

# Function to create or recreate a collection
def create_collection(collection_name="Chatbox"):
    try:
        # Get existing collections before creation
        collections = client.get_collections()
        print("Existing collections before creation:", collections)

        # Check if the collection exists by trying to fetch it
        try:
            client.get_collection(collection_name)
            print(f"Collection '{collection_name}' already exists.")
        except Exception:
            # Create collection if it doesn't exist
            client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=768,
                    distance=rest.Distance.COSINE
                )
            )
            print(f"Collection '{collection_name}' created.")

        # Get collections after creation
        collections = client.get_collections()
        print("Existing collections after creation:", collections)
            
    except Exception as e:
        print("An error occurred:", e)
        if "server engine not running" in str(e).lower():
            print("The Qdrant server engine is not running. Please check the server status.")

# Step 2: Function to insert vectors into the collection
def insert_vectors(collection_name="Chatbox", vectors_data=None):
    if vectors_data is None:
        # Sample vectors if none provided
        vectors_data = [
            {"id": 1, "vector": [0.1] * 768},
            {"id": 2, "vector": [0.2] * 768},
            {"id": 3, "vector": [0.3] * 768}
        ]
    
    try:
        client.upsert(
            collection_name=collection_name,
            points=vectors_data
        )
        print(f"{len(vectors_data)} vectors inserted into '{collection_name}' collection.")
    except Exception as e:
        print("An error occurred while inserting vectors:", e)

# Step 3: Function to search for similar vectors
def search_vector(collection_name="Chatbox", search_vector=None, top_k=5):
    if search_vector is None:
        # Use a sample vector if none provided
        search_vector = [0.1] * 768

    try:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=search_vector,
            limit=top_k
        )
        print(f"Top {top_k} search results for the input vector:", search_results)
    except Exception as e:
        print("An error occurred while searching:", e)

# Step 4: Check collections and points in the collection
def check_collections():
    try:
        collections = client.get_collections()
        print("Available collections:", collections)
    except Exception as e:
        print("An error occurred while fetching collections:", e)

def check_collection_details(collection_name="Chatbox"):
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Details of collection '{collection_name}':", collection_info)
    except Exception as e:
        print(f"An error occurred while fetching details of '{collection_name}':", e)

if __name__ == "__main__":
    # Step 1: Create Collection
    create_collection()

    # Step 2: Insert Vectors
    insert_vectors()

    # Step 3: Search Vector
    search_vector()

    # Step 4: Check Collections and Collection Details
    check_collections()
    check_collection_details()
