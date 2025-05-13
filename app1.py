import os
import io
import streamlit as st
import PyPDF2
import fitz  # PyMuPDF
import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusException

# Load environment variables
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Initialize OpenAI API key
MODEL_NAME = os.getenv("LLM_MODEL")
TEMPERATURE = 0.5
collection = None
json_file_path = 'uploaded_files.json'  # Define the JSON file path for storing uploaded filenames

# Streamlit configuration
st.set_page_config(page_title="PDF Content Analyzer")
st.header("PDF Content Analyzer")
st.sidebar.title("Options")

# Clear conversation
clear_button = st.sidebar.button("Clear Conversation", key="clear")
if clear_button or "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in content extraction and question answering."}
    ]

def format_chat_history(chat_history):
    formatted_history = ""
    for i, (question, answer) in enumerate(chat_history):
        formatted_history += f"Q{i+1}: {question}\nA{i+1}: {answer}\n"
    return formatted_history

# Upload PDF files
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Function to extract text from PDF by page
def extract_text_by_page(pdf_file):
    page_texts = []
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page_number in range(len(reader.pages)):
            page = reader.pages[page_number]
            text = page.extract_text()
            if text.strip():
                page_texts.append((page_number + 1, text))  # Store 1-indexed page number and text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return page_texts

# Function to chunk text by page
def chunk_text_by_page(page_texts, chunk_size=100):
    chunks = []
    chunk_page_numbers = []
    for page_number, text in page_texts:
        sentences = sent_tokenize(text)
        current_chunk = ''
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
                current_chunk += sentence + ' '
            else:
                chunks.append(current_chunk.strip())
                chunk_page_numbers.append(page_number)
                current_chunk = sentence + ' '
        if current_chunk:
            chunks.append(current_chunk.strip())
            chunk_page_numbers.append(page_number)
    return chunks, chunk_page_numbers

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

# Connect to MILVUS
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
            collection.create_index(field_name="embeddings", index_params={"index_type": "IVF_FLAT", "params": {"nlist": 1024}, "metric_type": "L2"})
            collection.load()
        else:
            collection = Collection("exmpcollection1")
        return collection
    except MilvusException as e:
        st.error(f"Failed to connect to Milvus: {e}")
        return None

# Function to clear the Milvus collection
def clear_milvus_collection(collection):
    if collection:
        try:
            collection.drop()
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(fields, "index1")
            collection = Collection("exmpcollection1", schema)
            collection.create_index(field_name="embeddings", index_params={"index_type": "IVF_FLAT", "params": {"nlist": 1024}, "metric_type": "L2"})
            collection.load()
        except MilvusException as e:
            st.error(f"Failed to clear Milvus collection: {e}")

# Initialize collection
collection = connect_to_milvus()

# Function to read filenames from a JSON file
def read_uploaded_files(json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            return json.load(file)
    return []

# Function to write filenames to a JSON file
def write_uploaded_files(json_file_path, file_list):
    with open(json_file_path, 'w') as file:
        json.dump(file_list, file, indent=4)

# Function to add a new filename to the JSON file
def add_uploaded_file(json_file_path, filename):
    file_list = read_uploaded_files(json_file_path)
    if filename not in file_list:
        file_list.append(filename)
    write_uploaded_files(json_file_path, file_list)

def clear_uploaded_files(json_file_path, collection):
    # Clear JSON file
    try:
        write_uploaded_files(json_file_path, [])
        st.sidebar.success("File history cleared from JSON file.")
    except Exception as e:
        st.error(f"Error clearing JSON file: {e}")

    # Clear Milvus collection
    clear_milvus_collection(collection)  

clear_DB_button = st.sidebar.button("Clear File History", key="ClearFileHistory")
if clear_DB_button:
    clear_uploaded_files(json_file_path, collection)
    st.experimental_rerun()  # Refresh the app to update the sidebar

# Sidebar: Display uploaded files
uploaded_files = read_uploaded_files(json_file_path)
st.sidebar.title("Uploaded Files")
if uploaded_files:
    for file in uploaded_files:
        st.sidebar.write(file)
else:
    st.sidebar.write("No files uploaded.")

# Process uploaded files
if pdf_files:
    try:
        all_embeddings = []
        page_data = []

        for pdf_file in pdf_files:
            # Extract text by page
            pdf_data = io.BytesIO(pdf_file.getvalue())
            page_texts = extract_text_by_page(pdf_data)
            pdf_name = pdf_file.name
            page_data.append((pdf_name, page_texts))
            # Add the uploaded file to the list and save it
            add_uploaded_file(json_file_path, pdf_name)

        for pdf_name, page_texts in page_data:
            chunks, chunk_page_numbers = chunk_text_by_page(page_texts)
            embeddings = embed_chunks(chunks)
            hierarchical_structure = recursive_clustering(embeddings, chunks, 2)
            all_embeddings.append((embeddings, chunks, chunk_page_numbers, pdf_name, hierarchical_structure))

        st.sidebar.header("Extracted Text from PDFs")
        for pdf_name, page_texts in page_data:
            st.sidebar.subheader(pdf_name)

        for embeddings, chunks, chunk_page_numbers, pdf_name, hierarchical_structure in all_embeddings:
            num_embeddings = embeddings.shape[0]
            metadata = [{"pdf_name": pdf_name, "chunk": chunk, "page_number": chunk_page_numbers[i], "chunk_index": i, "hierarchical_level": 1} for i, chunk in enumerate(chunks)]

            if num_embeddings != len(metadata):
                st.error(f"Mismatch in embeddings and metadata length: {num_embeddings} vs {len(metadata)}")
                continue  # Skip this batch if there's a mismatch

            # Include hierarchical summaries
            for level, summaries in hierarchical_structure.items():
                if isinstance(level, str) and "level_" in level:
                    level_num = int(level.split('_')[1])
                    for summary in summaries:
                        summary_embedding = embed_chunks([summary])[0]
                        embeddings = np.vstack([embeddings, summary_embedding])
                        metadata.append({"pdf_name": pdf_name, "chunk": summary, "page_number": None, "chunk_index": -1, "hierarchical_level": level_num})

            # Convert metadata to JSON
            metadata_json = [json.dumps(m) for m in metadata]

            # Insert data into Milvus
            try:
                insert_result = collection.insert([embeddings, metadata_json])
                collection.flush()
            except MilvusException as e:
                st.error(f"Failed to insert data into Milvus: {e}")

        collection.load()
        st.sidebar.success("Data inserted into Milvus successfully!")
    except Exception as e:
        st.error(f"Failed to process PDFs: {e}")

# Define retrieval functions using Milvus
def retrieve_documents(query_embedding, top_k=10):
    if not collection:
        return []

    # Search for similar embeddings in Milvus
    search_params = {"metric_type": "L2", "params": {"nprobe": 15}}
    results = collection.search(data=[query_embedding], anns_field="embeddings", param=search_params, limit=top_k, expr=None, output_fields=["metadata"])

    return results

# Function to create context from metadata
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

# Function to normalize text
def normalize_text(text):
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s*-\s*", "-", text)  # Remove spaces around hyphens
    return text.strip()

# Function to highlight text in the uploaded PDF
def highlight_text_in_pdf(uploaded_file, answer, context_metadata):
    # Open the uploaded PDF file
    document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    highlighted = False

    # Iterate through metadata and find matching text
    for meta in context_metadata:
        page_number = meta["page_number"]   # Adjust for 0-indexed page number
        chunk = meta["chunk"]

        if answer in chunk:
            # Open the specified page
            page = document.load_page(page_number)

            # Normalize the page text and search for the normalized version of the answer
            normalized_chunk = normalize_text(chunk)
            normalized_answer = normalize_text(answer)

            # Search for the answer in the chunk
            text_instances = page.search_for(normalized_answer)

            # Highlight all found instances
            if text_instances:
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.update()
                highlighted = True

    # Save the highlighted PDF
    if highlighted:
        pdf_bytes = io.BytesIO()
        document.save(pdf_bytes, garbage=4, deflate=True, clean=True)
        document.close()
        pdf_bytes.seek(0)
        return pdf_bytes
    else:
        document.close()
        return None

# Function to find answer using GPT
def find_answer_gpt(question, context):
    try:
        response = client.chat.completions.create(model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant who reads reports."},
            {"role": "user", "content": f"Context metadata contains up to 10 elements in JSON format. Each element contains pdf_name, chunk, page_number, and additional fields. Please answer the question precisely and accurately along with page numbers and filenames. If the query is from multiple pages tell all the page numbers. Consider multiple elements of contexts from same file as single context. Return answer as natural language, page number, and filename in json format\n\nContext metadata: {context_metadata}\n\nQuestion: {question}"}
        ],
        max_tokens=150,
        temperature=TEMPERATURE)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Error occurred."

# Streamlit interface for querying
query = st.text_input("Enter your query:", key="query_input")

# Process uploaded file and query
if query:
    query_embedding = embed_chunks([query])[0]
    results = retrieve_documents(query_embedding)

    if results:
        with st.spinner("Assistant is typing..."):
            context, context_metadata = create_context_from_metadata(results)
            st.write("Context Metadata:")
            st.write(context_metadata)

            if context_metadata:
                answer = find_answer_gpt(query, context_metadata)
                st.write(f"Answer: {answer}")


                # Highlight the answer in the uploaded PDF
                highlighted_pdf = highlight_text_in_pdf(pdf_file, answer, context_metadata)

                if highlighted_pdf:
                    st.download_button(
                        label="Download Highlighted PDF",
                        data=highlighted_pdf,
                        file_name=f"highlighted_{pdf_file.name}",
                        mime="application/pdf"
                    )
                else:
                    st.warning("No matching text found for highlighting.")
