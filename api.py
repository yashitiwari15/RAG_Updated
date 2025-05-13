import os
import io
import fitz  # PyMuPDF
import PyPDF2
import re
import streamlit as st
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import numpy as np
from sklearn.mixture import GaussianMixture
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusException
import json
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv(find_dotenv())

# Initialize OpenAI API key
MODEL_NAME = os.getenv("LLM_MODEL")
TEMPERATURE = 0.5

# Function to extract text from PDF by page and ensure patient name is stored in metadata
def extract_text_by_page_with_metadata(pdf_file):
    page_texts = []
    patient_name = None
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page_number in range(len(reader.pages)):
            page = reader.pages[page_number]
            text = page.extract_text()

            # Try to detect patient name on the page
            if not patient_name:
                patient_name = extract_patient_name(text)

            # Store the extracted text and associated metadata
            if text.strip():
                page_texts.append((page_number + 1, text, patient_name)) 
    except Exception as e:
        return str(e), None
    return page_texts, None

# Function to extract patient name from text
def extract_patient_name(text):
    name_match = re.search(r"Name:\s+([A-Za-z\s]+)", text)
    if name_match:
        return name_match.group(1).strip()
    return None

# Function to chunk text by page and include patient name in metadata
def chunk_text_by_page(page_texts, chunk_size=100):
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

# Function to embed chunks
def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
    return embeddings

# Function to summarize text using GPT
def summarize_text_gpt(text):
    try:
        response = client.chat.completions.create(model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
        ],
        max_tokens=150,
        temperature=0.5)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Error occurred."

# Recursive clustering and summarization
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

def connect_to_milvus():
    try:
        connections.connect("default", host="localhost", port="19530")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384),
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
        else:
            collection = Collection("exmpcollection1")
        return collection
    except MilvusException as e:
        st.error(f"Failed to connect to Milvus: {e}")
        return None

collection = connect_to_milvus()
if collection is None:
    st.error("Failed to connect to Milvus. Exiting...")
    st.stop()


def read_uploaded_files(json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []
    return []

def write_uploaded_files(json_file_path, file_list):
    with open(json_file_path, 'w') as file:
        json.dump(file_list, file, indent=4)

# Function to add a new filename to the JSON file
def add_uploaded_file(json_file_path, filename):
    file_list = read_uploaded_files(json_file_path)
    if filename not in file_list:
        file_list.append(filename)
    write_uploaded_files(json_file_path, file_list)

# Define retrieval functions using Milvus
def retrieve_documents(query_embedding, top_k=10):
    if not collection:
        return []

    search_params = {"metric_type": "COSINE", "params": {"ef": 200}}
    try:
        results = collection.search(
            data=[query_embedding],
            anns_field="embeddings",
            param=search_params,
            limit=top_k,
            expr=None,
            output_fields=["metadata"]
        )
        return results
    except MilvusException as e:
        st.error(f"Error during document retrieval: {e}")
        return []


def create_context_from_metadata(results):
    context_chunks = []
    context_metadata = []

    for result in results:
        for hit in result:
            metadata_str = hit.entity.get("metadata")
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
            chunk = metadata.get("chunk", "")
            context_chunks.append(chunk)
            context_metadata.append(metadata)

    context = " ".join(context_chunks)
    return context, context_metadata

def normalize_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return text.strip()

def highlight_pdf_text(pdf_path, highlight_text, page_numbers):
    try:
        document = fitz.open(pdf_path)
        for page_num in page_numbers:
            page = document.load_page(page_num - 1)
            text_instances = page.search_for(highlight_text)
            if text_instances:
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.update()

        downloads_path = Path.home() / "Downloads" / f"highlighted_{Path(pdf_path).name}"
        document.save(downloads_path)
        document.close()

        return str(downloads_path), None

    except Exception as e:
        return None, str(e)

def parse_gpt_response(response):
    try:
        data = json.loads(response)
        return data
    except json.JSONDecodeError as je:
        return None, str(je)
    except Exception as e:
        return None, str(e)

def get_next_file_path(base_dir, base_filename, file_extension="pdf"):
    base_path = Path(base_dir) / base_filename
    count = 1
    while (base_path.with_stem(f"{base_filename}{count}").with_suffix(f".{file_extension}")).exists():
        count += 1
    return base_path.with_stem(f"{base_filename}{count}").with_suffix(f".{file_extension}")

# Function to interact with GPT-3.5 for answering questions based on context
def find_answer_gpt(question, context_metadata):
    try:
        context_info = json.dumps(context_metadata, indent=2)

        response = client.chat.completions.create(model=MODEL_NAME,
        messages=[os.getenv("PROMPT") 
        ],
        max_tokens=200,
        temperature=TEMPERATURE)

        raw_response = response.choices[0].message.content.strip()
        return raw_response

    except Exception as e:
        return f"Error occurred in find_answer_gpt: {e}"

# Define API endpoints
@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            pdf_data = io.BytesIO(file.read())
            page_texts, error = extract_text_by_page_with_metadata(pdf_data)
            if error:
                return jsonify({"error": error}), 500
            filename = file.filename
            chunks, chunk_page_numbers, patient_names = chunk_text_by_page(page_texts)
            embeddings = embed_chunks(chunks)
            hierarchical_structure = recursive_clustering(embeddings, chunks, 2)

            metadata = [{"pdf_name": filename, "chunk": chunk, "page_number": chunk_page_numbers[i], "patient_name": patient_names[i], "chunk_index": i, "hierarchical_level": 1} for i, chunk in enumerate(chunks)]

            for level, summaries in hierarchical_structure.items():
                if isinstance(level, str) and "level_" in level:
                    level_num = int(level.split('_')[1])
                    for summary in summaries:
                        summary_embedding = embed_chunks([summary])[0]
                        embeddings = np.vstack([embeddings, summary_embedding])
                        metadata.append({"pdf_name": filename, "chunk": summary, "page_number": None, "patient_name": patient_names, "chunk_index": -1, "hierarchical_level": level_num})

            metadata_json = [json.dumps(m) for m in metadata]
            insert_result = collection.insert([embeddings, metadata_json])
            collection.flush()

            return jsonify({"message": "File uploaded and processed successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["POST"])
def query_pdf():
    try:
        print("Into my query API X")
        data = request.get_json()
        print("Into my query API 1 ", data)
        query = data.get("query", "")
        print("Into my query API 2", query)
        query_embedding = embed_chunks([query])[0]
        results = retrieve_documents(query_embedding)
        print("Into my query API 1 ", results)
        if not results:
            return jsonify({"error": "No results found"}), 404
        context, context_metadata = create_context_from_metadata(results)
        if not context_metadata:
            return jsonify({"error": "No context metadata found"}), 404
        print("Going to find answer from gpt")
        answer = find_answer_gpt(query, context_metadata)
        print("Found an answer: ", answer)
        return answer
        # parsed_response, error = parse_gpt_response(answer)
        # if error:
        #     return jsonify({"error": error}), 500

        # highlight_text = parsed_response.get("highlight", "")
        # filename = parsed_response.get("filename", "")
        # page_numbers = parsed_response.get("page_number", [])

        # pdf_path = Path("/Users/sarthakgarg/Documents/Sarthak/") / filename
        # output_path = get_next_file_path(Path.home() / "Downloads", "highlighted_output")

        # highlighted_pdf, error = highlight_pdf_text(pdf_path, highlight_text, page_numbers)
        # if error:
        #     return jsonify({"error": error}), 500

        # return jsonify({"highlighted_pdf": str(output_path)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
