from datetime import datetime
from pymilvus import connections, Collection, utility, MilvusException
from flask import Blueprint, request, jsonify
import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from sentence_transformers import SentenceTransformer, CrossEncoder
import nltk
from nltk.tokenize import word_tokenize
import json
import os
from datetime import datetime

# Ensure you have downloaded necessary NLTK data
nltk.download('punkt')

query_bp = Blueprint('query_bp', __name__)

MODEL_NAME = os.getenv("LLM_MODEL")
TEMPERATURE = 0.5

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

        # Generate the answer using OpenAI GPT, extract only highlight and ans
        result = find_answer_gpt(query, context_metadata, previous_conversation)

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

import json

# def find_answer_gpt(question, context_metadata, previous_conversation):
#     # Convert context metadata to JSON format
#     context_info = json.dumps(context_metadata, indent=2)
#     print("this is context metadata",context_info)

#     # Call the OpenAI API for LLM assistance with chat history
#     response = openai.ChatCompletion.create(
#         model=MODEL_NAME,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", 
#              "content": f"""Please answer the following question by returning only a JSON response. The JSON response should include the following fields: 'highlight', 'filename', 'page_number', and 'ans'. Use this format:
#             Each highlight must correspond exactly to the text in the 'original_text' field of the context_metadata. **Strictly avoid using the 'summary_text' field in the metadata for highlights.** The 'summary_text' can be used for crafting the 'ans' field only.
#             Make sure that the response does not contain any additional markers such as '''json or '''.
#             The JSON response should include the following fields: 'highlight', 'filename', 'page_number', and 'ans'. Use this format:
# {{
# "highlight": [
#     {{
#       "text": "[Need exact text to be highlighted from original_text field in context_metadata. Strictly do not use the summary_text field of context_metadata for highlights. Each highlight must correspond to the part of the text from the original_text only.]",
#       "page_number": [Page number of the highlighted text],
#       "filename": "[PDF file name where this highlighted text is located]"
#     }},
#     [Repeat for each highlight from different files and page numbers]

# ],
# "ans": "[Give a natural language according to the query from the context_metadata]"
# }}

# Conversation so far: {previous_conversation}

# Answer the following question: {question}

# Context metadata: {context_info}"""}
#         ],
#         max_tokens=1000,
#         temperature=TEMPERATURE
#     )

#     # Extract the raw response
#     raw_response = response.choices[0].message['content'].strip()
#     highlight_data = json.loads(raw_response)
#     #print(f"Raw Response: {raw_response}")  # Print the raw response for debugging

#     # Remove '''json and ''' from the response if they exist
#     if raw_response.startswith("'''json"):
#         raw_response = raw_response[7:]
#     if raw_response.endswith("'''"):
#         raw_response = raw_response[:-3]

#     # Log the cleaned response


#     # Try to parse the cleaned response as JSON
#     try:
#         gpt_answer = json.loads(raw_response)  # Attempt to parse as JSON
#     except json.JSONDecodeError as e:
#         # Log the specific error and return
#         print(f"JSONDecodeError: {str(e)}")
#         return {"error": f"Failed to parse GPT response as valid JSON. Error: {str(e)}"}

#     # Safely extract "highlight" and "ans" fields, checking if they exist
#     highlight = gpt_answer.get("highlight", None)
#     ans = gpt_answer.get("ans", None)

#     valid_highlights = []
#     for highlight in gpt_answer.get("highlight", []):
#         for item in context_metadata:
#             if (
#                 highlight["filename"] == item["pdf_name"] and
#                 highlight["page_number"] == item["page_number"] and
#                 highlight["text"] in item.get("original_text", "")
#             ):
#                 valid_highlights.append(highlight)

#     # Return the valid highlights and the answer
#     return {
#         "highlight": valid_highlights,
#         "ans": gpt_answer.get("ans", None)
#     }
#     # # Handle missing keys
#     # if highlight is None or ans is None:
#     #     return {"error": "Missing key: 'highlight' or 'ans'", "raw_response": raw_response}

#     # # Return only the "highlight" and "ans" fields
#     # return {
#     #     "highlight": highlight,
#     #     "ans": ans
#     # }

def find_answer_gpt(question, context_metadata, previous_conversation):
    # Convert context metadata to JSON format
    context_info = json.dumps(context_metadata, indent=2)
    print("this is context metadata", context_info)

    msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", 
             "content": f"""Please answer the following question by returning only a JSON response. 
            Make sure that the response does not contain any additional markers such as '''json or '''.
            In Highlight array, return only those relevant sections of context that has 'summary_text' as null and are used to form answer.
            The JSON response should include the following fields: 'highlight', 'filename', 'page_number', and 'ans'. Use this format:
{{
"highlight": [
    {{
      "text": "[Return orignal_text shortened by truncating but retaining essence of the text.]",
      "page_number": [Page number of the original_text],
      "filename": "[PDF file name where this original_text]"
      "source":[specify whether text is extracted from 'orignal_text' or 'summary_text' field of context_metadata]
    }},
    [Repeat for each highlight from different files and page numbers]
    
],
"ans": "[Give a natural language according to the query from the context_metadata]"
}}

Conversation so far: {previous_conversation}

Answer the following question: {question}

Context metadata: {context_info}"""}
        ]
    print("AG Message Begin: ", msgs, " :AG Messsage ENDS")
    # Call the OpenAI API for LLM assistance with chat history
    response = client.chat.completions.create(model=MODEL_NAME,
    messages=msgs,
    max_tokens=1000,
    temperature=TEMPERATURE)

    # Extract the raw response
    raw_response = response.choices[0].message.content.strip()

    # Try to parse the raw response as JSON
    try:
        gpt_answer = json.loads(raw_response)  # Attempt to parse as JSON
    except json.JSONDecodeError as e:
        # Log the specific error and return
        print(f"JSONDecodeError: {str(e)}")
        return {"error": f"Failed to parse GPT response as valid JSON. Error: {str(e)}"}

    # Safely extract "highlight" and "ans" fields, checking if they exist
    highlights = gpt_answer.get("highlight", [])
    ans = gpt_answer.get("ans", None)

    print("This is responce message",response)
    valid_highlights = []
    for highlight in highlights:
        for item in context_metadata:
            if (
                highlight["filename"] == item["pdf_name"] and
                highlight["page_number"] == item["page_number"] and
                highlight["text"] in item.get("original_text", "")
            ):
                valid_highlights.append(highlight)

    # Return the valid highlights and the answer
    return {
        "highlight": highlights,
        "ans": ans
    }

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
