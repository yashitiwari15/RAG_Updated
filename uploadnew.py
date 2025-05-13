from flask import Blueprint, request, jsonify
import io
import json
import openai
import re
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusException
import PyPDF2
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.mixture import GaussianMixture

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI for Azure (for v0.28.x)
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_ENDPOINT")
openai.api_version = os.getenv("API_VERSION")
openai.api_key = os.getenv("AZURE_API_KEY")

upload_bp = Blueprint('upload_bp', __name__)

def connect_to_milvus():
    try:
        connections.connect("default", host="0.0.0.0", port="19530")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="pdf_name", dtype=DataType.VARCHAR, max_length=255),  # Scalar field for the filename
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields, "index1")

        if not utility.has_collection("exmpcollection1"):
            collection = Collection("exmpcollection1", schema)
            index_params = {
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
                "metric_type": "COSINE"
            }
            collection.create_index(field_name="embeddings", index_params=index_params)
            collection.load()
            print("Collection 'exmpcollection1' created successfully.")
        else:
            collection = Collection("exmpcollection1")
            print("Collection 'exmpcollection1' loaded successfully.")
        
        return collection
    except MilvusException as e:
        print(f"Failed to connect to Milvus or create the collection: {e}")
        return None

collection = connect_to_milvus()

def check_schema(collection_name):
    try:
        connections.connect("default", host="localhost", port="19530")
        collection = Collection(collection_name)
        schema = collection.schema
        print("Schema Fields:")
        for field in schema.fields:
            print(f" - {field.name}: {field.dtype}")
    except Exception as e:
        print(f"Failed to check schema: {e}")

def drop_collection(collection_name):
    try:
        connections.connect("default", host="localhost", port="19530")
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"Collection '{collection_name}' dropped successfully.")
        else:
            print(f"Collection '{collection_name}' does not exist.")
    except Exception as e:
        print(f"Failed to drop collection '{collection_name}': {e}")

drop_collection("exmpcollection1")  # Drop the collection (only if it exists)

@upload_bp.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            filename = file.filename.lower()  # Convert filename to lowercase
            
            if collection is None:
                return jsonify({"error": "Failed to initialize collection."}), 500

            # Check for duplicate
            filter_expr = f'pdf_name == "{filename}"'
            existing_records = collection.query(
                expr=filter_expr,
                output_fields=["pdf_name"]
            )
            if existing_records:
                print("Duplicate file detected. Aborting upload.")
                return jsonify({"error": "This file has already been uploaded."}), 400

            pdf_data = io.BytesIO(file.read())
            page_texts, error = extract_text_by_page_with_metadata(pdf_data)
            if error:
                return jsonify({"error": error}), 500
            
            chunks, chunk_page_numbers, patient_names = chunk_text_by_page(page_texts)
            embeddings = embed_chunks(chunks)
            hierarchical_structure = recursive_clustering(embeddings, chunks, 2)
            print("this is the summary", hierarchical_structure)

            metadata = [{"pdf_name": filename, "chunk": chunk, "page_number": chunk_page_numbers[i], "patient_name": patient_names[i], "chunk_index": i, "hierarchical_level": 1} for i, chunk in enumerate(chunks)]

            # Prepare data to insert
            metadata_json = [json.dumps(m) for m in metadata]
            pdf_names = [filename] * len(embeddings)

            collection.insert([embeddings, pdf_names, metadata_json])
            collection.flush()

            return jsonify({"message": "File uploaded and processed successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

def extract_text_by_page_with_metadata(pdf_file):
    page_texts = []
    patient_name = None
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page_number in range(len(reader.pages)):
            page = reader.pages[page_number]
            text = page.extract_text()
            if not patient_name:
                patient_name = extract_patient_name(text)
            if text.strip():
                page_texts.append((page_number + 1, text, patient_name)) 
    except Exception as e:
        return str(e), None
    return page_texts, None

def extract_patient_name(text):
    name_match = re.search(r"Name:\s+([A-Za-z\s]+)", text)
    if name_match:
        return name_match.group(1).strip()
    return None

def chunk_text_by_page(page_texts, chunk_size=200):
    chunks = []
    chunk_page_numbers = []
    patient_names = []
    for page_number, text, patient_name in page_texts:
        sentences = sent_tokenize(text)
        current_chunk = ''
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
                current_chunk += sentence + ' '
            else:
                chunks.append(current_chunk.strip())
                chunk_page_numbers.append(page_number)
                patient_names.append(patient_name)
                current_chunk = sentence + ' '
        if current_chunk:
            chunks.append(current_chunk.strip())
            chunk_page_numbers.append(page_number)
            patient_names.append(patient_name)
    return chunks, chunk_page_numbers, patient_names

def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
    return embeddings

def recursive_clustering(embeddings, chunks, depth=2, current_depth=1):
    if current_depth > depth or len(chunks) < 2:
        return {f'level_{current_depth}': chunks}
    
    gmm = GaussianMixture(n_components=2, covariance_type='tied')
    gmm.fit(embeddings)
    cluster_labels = gmm.predict(embeddings)
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(chunks[idx])

    summaries = {}
    for cluster_id, cluster_chunks in clusters.items():
        combined_text = ' '.join(cluster_chunks)
        summary = summarize_text_gpt(combined_text)
        summary_embedding = embed_chunks([summary])
        summaries[cluster_id] = recursive_clustering(summary_embedding, [summary], depth, current_depth + 1)

    return summaries

def summarize_text_gpt(text):
    try:
        response = openai.ChatCompletion.create(
            engine=os.getenv("LLM_MODEL"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ],
            max_tokens=1000,
            temperature=float(os.getenv("TEMPERATURE"))
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error occurred: {e}"
