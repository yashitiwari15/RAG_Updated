from azure.cosmos import CosmosClient, PartitionKey
from sentence_transformers import SentenceTransformer
import numpy as np
from config import settings  # Import settings from config.py
from dotenv import load_dotenv

# Load environment variables (if necessary)
load_dotenv()

# Initialize Cosmos DB client using the credentials from config.py
client = CosmosClient('https://cosmosdb-001.documents.azure.com:443/', 'dbi3fM20YOzPIuTBlAkFwdLeEiJCjFZuE8kCBhE7NH3H9YswoO9PYueBaYXj60mdMKJICw2beC9yACDbix2GLA==')
# Connect to the database and container using settings from config.py
database = client.get_database_client(settings['database_id'])  # Use the database_id from config.py
container = database.get_container_client(settings['container_id'])  # Use the container_id from config.py

# Function to embed query text
def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()  # Convert embeddings to numpy arrays
    return embeddings

def query_cosmos_by_vector(query_text):
    # Step 1a: Generate the query vector embedding
    query_embedding = embed_chunks([query_text])[0]  # Generate the embedding for the query text

    # Step 1b: Fetch all items from Cosmos DB
    query = "SELECT * FROM c"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))

    # Step 1c: Calculate cosine similarity between query and each stored chunk
    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = []
    for item in items:
        # Step 1d: Check if 'embedding' exists in the item
        if 'embedding' not in item:
            print(f"Skipping item {item['id']} because it does not have an embedding.")
            continue
        
        stored_embedding = np.array(item['embedding'])  # Convert the stored embedding back to numpy array
        similarity = cosine_similarity(query_embedding, stored_embedding)
        similarities.append((item['chunk'], similarity))

    # Step 1e: Sort by similarity and return the most relevant chunks
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:5]  # Return top 5 similar chunks

# Example of querying the database with some text
query_text = "What is the oxygen saturation?"
top_matches = query_cosmos_by_vector(query_text)

for chunk, similarity in top_matches:
    print(f"Similarity: {similarity:.4f}, Chunk: {chunk}")
