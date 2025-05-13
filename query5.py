# import sys
# import os
# from datetime import datetime
# from pymilvus import connections, Collection, utility, MilvusException
# from flask import Blueprint, request, jsonify
# import json
# from sentence_transformers import SentenceTransformer, CrossEncoder
# import nltk
# from nltk.tokenize import word_tokenize
# from dotenv import load_dotenv

# # Add the parent directory of 'STEPSAI_PROJECT' to the Python path
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# # Import the LLMFactory from the llm_package inside STEPSAI_PROJECT
# from llm_package import LLMFactory

# # Load environment variables from .env file
# load_dotenv()

# # Ensure you have downloaded necessary NLTK data
# nltk.download('punkt')

# query_bp = Blueprint('query_bp', __name__)

# # Initialize the Cross-Encoder model for re-ranking
# cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# # In-memory store for chat history
# chat_history = {}

# def connect_to_milvus():
#     connections.connect("default", host="0.0.0.0", port="19530")
#     if utility.has_collection("exmpcollection1"):
#         return Collection("exmpcollection1")
#     return None

# collection = connect_to_milvus()

# if collection is None:
#     raise Exception("Failed to connect to Milvus. Exiting...", 400)

# @query_bp.route("/query", methods=["POST"])
# def query_pdf():
#     try:
#         data = request.get_json()
#         user_id = data.get("user_id", "default")
#         query = data.get("query", "")
#         filenames = data.get("filenames", None)

#         if not query:
#             return jsonify({"error": "Query parameter is missing"}), 400

#         if user_id not in chat_history:
#             chat_history[user_id] = []

#         query_embedding = embed_chunks([query])[0]
#         results = retrieve_documents(query_embedding, query, filenames=filenames)
#         if not results:
#             return jsonify({"error": "No results found"}), 404

#         reranked_results = cross_encoder_rerank(query, results)
#         context, context_metadata = create_context_from_metadata(reranked_results)
#         if not context_metadata:
#             return jsonify({"error": "No context metadata found"}), 404

#         previous_conversation = "\n".join(
#             f"User: {entry['query']}\nAssistant: {entry['answer']}"
#             for entry in chat_history[user_id]
#         )

#         # Get the LLM client using the factory
#         llm_client = LLMFactory.get_llm_client()

#         # Generate the answer using the LLM client
#         result = llm_client.find_answer_gpt(query, context_metadata, previous_conversation)

#         chat_history[user_id].append({
#             "query": query,
#             "answer": result['ans']
#         })

#         save_interaction_to_json(user_id, query, result['ans'])

#         return jsonify(result)

#     except MilvusException as e:
#         return jsonify(f"Milvus Error: {e}")
#     except KeyError as e:
#         return jsonify({"error": f"Missing key: {e}"}), 400
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# def embed_chunks(chunks):
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
#     return embeddings

# # Other helper functions remain unchanged

# def retrieve_documents(query_embedding, query, filenames=None, top_k=15):
#     if not collection:
#         return []

#     search_params = {"metric_type": "COSINE", "params": {"ef": 200}}

#     if filenames:
#         filenames_expr = f"pdf_name in [{', '.join([f'\"{filename}\"' for filename in filenames])}]"
#         expr = filenames_expr
#     else:
#         expr = None

#     results = collection.search(
#         data=[query_embedding],
#         anns_field="embeddings",
#         param=search_params,
#         limit=top_k,
#         expr=expr,
#         output_fields=["metadata"]
#     )

#     # Get the chunks and their metadata from the results
#     retrieved_chunks = []
#     for result in results:
#         for hit in result:
#             metadata_str = hit.entity.get("metadata")
#             metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
#             chunk = metadata.get("chunk", "")
#             retrieved_chunks.append((chunk, metadata))  # Store both chunk and metadata

#     return retrieved_chunks

# def cross_encoder_rerank(query, retrieved_chunks):
#     cross_encoder_input = [(query, chunk) for chunk, _ in retrieved_chunks]

#     # Get relevance scores from the Cross-Encoder model
#     scores = cross_encoder_model.predict(cross_encoder_input)

#     # Combine chunks, metadata, and scores
#     chunks_with_scores = [(chunk, metadata, score) for (chunk, metadata), score in zip(retrieved_chunks, scores)]

#     # Sort by score in descending order
#     ranked_chunks = sorted(chunks_with_scores, key=lambda x: x[2], reverse=True)

#     # Return the top 5 ranked chunks with their metadata
#     return [(chunk, metadata) for chunk, metadata, _ in ranked_chunks[:5]]

# def create_context_from_metadata(reranked_results):
#     context_chunks = []
#     context_metadata = []

#     for chunk, metadata in reranked_results:
#         context_chunks.append(chunk)
#         context_metadata.append(metadata)

#     context = " ".join(context_chunks)
#     return context, context_metadata

# def save_interaction_to_json(user_id, question, answer):
#     # Path to interactions.json file
#     output_dir = "interactions"
#     filepath = os.path.join(output_dir, "interactions.json")

#     # Check if the file exists, and load existing interactions
#     if os.path.exists(filepath):
#         with open(filepath, 'r') as json_file:
#             try:
#                 interactions = json.load(json_file)
#             except json.JSONDecodeError:
#                 interactions = []  # Handle case where file is empty or corrupted
#     else:
#         interactions = []

#     # Check if there is already an entry for this user
#     for interaction in interactions:
#         if interaction["user_id"] == user_id:
#             # Avoid adding duplicate questions in the chat history
#             for chat in interaction["chat_history"]:
#                 if chat["query"] == question:
#                     return  # If the question already exists in chat history, do nothing

#             # Append the new question-answer to the chat history
#             interaction["chat_history"].append({
#                 "query": question,
#                 "answer": answer
#             })
#             break
#     else:
#         # If the user_id is not found, create a new entry for this user
#         interactions.append({
#             "timestamp": datetime.now().isoformat(),
#             "user_id": user_id,
#             "question": question,
#             "answer": answer,
#             "chat_history": [{
#                 "query": question,
#                 "answer": answer
#             }]
#         })

#     # Write the updated interactions back to the JSON file
#     with open(filepath, 'w') as json_file:
#         json.dump(interactions, json_file, indent=2)
from datetime import datetime
from pymilvus import connections, Collection, utility, MilvusException
from flask import Blueprint, request, jsonify
import json
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import nltk
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure you have downloaded necessary NLTK data
nltk.download('punkt')

# Import the LLMFactory from the new llm_package
from llm_package import LLMFactory

query_bp = Blueprint('query_bp', __name__)

# Initialize the Cross-Encoder model for re-ranking
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# In-memory store for chat history
chat_history = {}

def connect_to_milvus():
    connections.connect("default", host="0.0.0.0", port="19530")
    if utility.has_collection("exmpcollection1"):
        return Collection("exmpcollection1")
    return None

collection = connect_to_milvus()

if collection is None:
    raise Exception("Failed to connect to Milvus. Exiting...", 400)

@query_bp.route("/query", methods=["POST"])
def query_pdf():
    try:
        data = request.get_json()
        user_id = data.get("user_id", "default")  # Get user ID for personalized history
        query = data.get("query", "")
        filenames = data.get("filenames", None)  # Get the filenames from the request data

        if not query:
            return jsonify({"error": "Query parameter is missing"}), 400  # Bad Request

        # Initialize chat history for the user if not already present
        if user_id not in chat_history:
            chat_history[user_id] = []

        query_embedding = embed_chunks([query])[0]

        # Initial retrieval from Milvus (BM25 or vector-based search)
        results = retrieve_documents(query_embedding, query, filenames=filenames)
        if not results:
            return jsonify({"error": "No results found"}), 404

        # Re-rank using Cross-Encoder
        reranked_results = cross_encoder_rerank(query, results)

        context, context_metadata = create_context_from_metadata(reranked_results)
        if not context_metadata:
            return jsonify({"error": "No context metadata found"}), 404

        # Build the previous conversation from the user's chat history
        previous_conversation = "\n".join(
            f"User: {entry['query']}\nAssistant: {entry['answer']}"
            for entry in chat_history[user_id]
        )

        # Get the LLM client using the factory
        llm_client = LLMFactory.get_llm_client()

        # Generate the answer using the LLM client
        result = llm_client.find_answer_gpt(query, context_metadata, previous_conversation)

        # Append the current query and answer to chat history (only question and answer)
        chat_history[user_id].append({
            "query": query,
            "answer": result['ans']  # Only store the final answer in the chat history
        })

        # Save the interaction to the JSON file (with full chat history)
        save_interaction_to_json(user_id, query, result['ans'])

        # Return only the highlight and ans in the response
        return jsonify(result)

    except MilvusException as e:
        return jsonify(f"Milvus Error: {e}")
    except KeyError as e:
        return jsonify({"error": f"Missing key: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
    return embeddings

def retrieve_documents(query_embedding, query, filenames=None, top_k=15):
    if not collection:
        return []

    search_params = {"metric_type": "COSINE", "params": {"ef": 200}}

    if filenames:
        filenames_expr = f"pdf_name in [{', '.join([f'\"{filename}\"' for filename in filenames])}]"
        expr = filenames_expr
    else:
        expr = None

    results = collection.search(
        data=[query_embedding],
        anns_field="embeddings",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["metadata"]
    )

    # Get the chunks and their metadata from the results
    retrieved_chunks = []
    for result in results:
        for hit in result:
            metadata_str = hit.entity.get("metadata")
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
            chunk = metadata.get("chunk", "")
            retrieved_chunks.append((chunk, metadata))  # Store both chunk and metadata

    return retrieved_chunks

def cross_encoder_rerank(query, retrieved_chunks):
    cross_encoder_input = [(query, chunk) for chunk, _ in retrieved_chunks]

    # Get relevance scores from the Cross-Encoder model
    scores = cross_encoder_model.predict(cross_encoder_input)

    # Combine chunks, metadata, and scores
    chunks_with_scores = [(chunk, metadata, score) for (chunk, metadata), score in zip(retrieved_chunks, scores)]

    # Sort by score in descending order
    ranked_chunks = sorted(chunks_with_scores, key=lambda x: x[2], reverse=True)

    # Return the top 5 ranked chunks with their metadata
    return [(chunk, metadata) for chunk, metadata, _ in ranked_chunks[:5]]

def create_context_from_metadata(reranked_results):
    context_chunks = []
    context_metadata = []

    for chunk, metadata in reranked_results:
        context_chunks.append(chunk)
        context_metadata.append(metadata)

    context = " ".join(context_chunks)
    return context, context_metadata

def save_interaction_to_json(user_id, question, answer):
    # Path to interactions.json file
    output_dir = "interactions"
    filepath = os.path.join(output_dir, "interactions.json")

    # Check if the file exists, and load existing interactions
    if os.path.exists(filepath):
        with open(filepath, 'r') as json_file:
            try:
                interactions = json.load(json_file)
            except json.JSONDecodeError:
                interactions = []  # Handle case where file is empty or corrupted
    else:
        interactions = []

    # Check if there is already an entry for this user
    for interaction in interactions:
        if interaction["user_id"] == user_id:
            # Avoid adding duplicate questions in the chat history
            for chat in interaction["chat_history"]:
                if chat["query"] == question:
                    return  # If the question already exists in chat history, do nothing

            # Append the new question-answer to the chat history
            interaction["chat_history"].append({
                "query": question,
                "answer": answer
            })
            break
    else:
        # If the user_id is not found, create a new entry for this user
        interactions.append({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "chat_history": [{
                "query": question,
                "answer": answer
            }]
        })

    # Write the updated interactions back to the JSON file
    with open(filepath, 'w') as json_file:
        json.dump(interactions, json_file, indent=2)

