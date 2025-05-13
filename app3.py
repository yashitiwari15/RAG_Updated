import os
import io
import streamlit as st
import fitz  # PyMuPDF
import PyPDF2
import re
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

# Load environment variables
load_dotenv(find_dotenv())

# Initialize OpenAI API key
MODEL_NAME = os.getenv("LLM_MODEL")
TEMPERATURE = 0.5
counter = 0
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
        st.error(f"Error extracting text from PDF: {e}")
    return page_texts

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
    print("Log recursive clustering ")
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
    print("Connction to milvus log")
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
            collection.create_index(field_name="embeddings", index_params={"index_type": "IVF_FLAT", "params": {"nlist": 1024}, "metric_type": "COSINE"})
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
    try:
        write_uploaded_files(json_file_path, [])
        st.sidebar.success("File history cleared from JSON file.")
    except Exception as e:
        st.error(f"Error clearing JSON file: {e}")
    clear_milvus_collection(collection)  

# Sidebar: Display uploaded files
json_file_path = 'uploaded_files.json'  # Define the JSON file path for storing uploaded filenames

clear_DB_button = st.sidebar.button("Clear File History", key="ClearFileHistory")
if clear_DB_button:
    clear_uploaded_files(json_file_path, collection)
    st.experimental_rerun()

uploaded_files = read_uploaded_files(json_file_path)
st.sidebar.title("Uploaded Files")
if uploaded_files:
    for file in uploaded_files:
        st.sidebar.write(file)
else:
    st.sidebar.write("No files uploaded.")

# Upload PDF files
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Process uploaded files
if pdf_files:
    st.write("Going to Query Embedding")
    st.write("Going to Query Embedding")

    try:
        all_embeddings = []
        page_data = []

        for pdf_file in pdf_files:
            st.write("Processing filename: ", pdf_file)
            pdf_data = io.BytesIO(pdf_file.getvalue())
            page_texts = extract_text_by_page_with_metadata(pdf_data)
            pdf_name = pdf_file.name
            page_data.append((pdf_name, page_texts))
            add_uploaded_file(json_file_path, pdf_name)

            for pdf_name, page_texts in page_data:
                chunks, chunk_page_numbers, patient_names = chunk_text_by_page(page_texts)
                embeddings = embed_chunks(chunks)
                hierarchical_structure = recursive_clustering(embeddings, chunks, 2)
                all_embeddings.append((embeddings, chunks, chunk_page_numbers, pdf_name, hierarchical_structure))

            st.sidebar.header("Extracted Text from PDFs")
            for pdf_name, page_texts in page_data:
                st.sidebar.subheader(pdf_name)

            for embeddings, chunks, chunk_page_numbers, pdf_name, hierarchical_structure in all_embeddings:
                num_embeddings = embeddings.shape[0]
                metadata = [{"pdf_name": pdf_name, "chunk": chunk, "page_number": chunk_page_numbers[i], "patient_name": patient_names[i], "chunk_index": i, "hierarchical_level": 1} for i, chunk in enumerate(chunks)]

                if num_embeddings != len(metadata):
                    st.error(f"Mismatch in embeddings and metadata length: {num_embeddings} vs {len(metadata)}")
                    continue

                for level, summaries in hierarchical_structure.items():
                    if isinstance(level, str) and "level_" in level:
                        level_num = int(level.split('_')[1])
                        for summary in summaries:
                            summary_embedding = embed_chunks([summary])[0]
                            embeddings = np.vstack([embeddings, summary_embedding])
                            metadata.append({"pdf_name": pdf_name, "chunk": summary, "page_number": None, "patient_name": patient_names, "chunk_index": -1, "hierarchical_level": level_num})

                metadata_json = [json.dumps(m) for m in metadata]

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

    search_params = {"metric_type": "COSINE", "params": {"ef":200}}
    results = collection.search(data=[query_embedding], anns_field="embeddings", param=search_params, limit=top_k, expr=None, output_fields=["metadata"])
    return results

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

        # Save the modified PDF to the Downloads directory
        downloads_path = Path.home() / "Downloads" / f"highlighted_{Path(pdf_path).name}"
        document.save(downloads_path)
        document.close()

        return downloads_path

    except Exception as e:
        print(f"Error in highlight_pdf_text: {e}")
        return None

def get_next_file_path(base_dir, base_filename, file_extension="pdf"):
    """
    This function returns the next available file path with an incrementing number.
    """
    base_path = Path(base_dir) / base_filename
    count = 1
    while (base_path.with_stem(f"{base_filename}{count}").with_suffix(f".{file_extension}")).exists():
        count += 1
    return base_path.with_stem(f"{base_filename}{count}").with_suffix(f".{file_extension}")
print("Lag testing abc")

# Replace the following line with your base directory and filename
base_directory = Path.home() / "Downloads"  # Set your desired directory
base_filename = "highlighted_output"



def find_answer_gpt(question, context_metadata):
    try:
        context_info = json.dumps(context_metadata, indent=2)

        response = client.chat.completions.create(model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant who reads reports."},
            {"role": "user",
             "content": f"""Please answer the following question by returning only a JSON response. The JSON response should include the following fields: 'highlight', 'filename', 'page_number', and 'ans'. Use this format:

{{
  "highlight": "[Exact text to be highlighted]",
  "filename": "[Name of the PDF file]",
  "page_number": [Page number(s)],
  "ans": "[Direct answer to the query]"
}}
You are given 10 metadata contaning feilds like chunk and patient name. Answer the question by corretly matching the name in query and patient_name in metadata only.
Ensure the JSON response is valid and all fields are correctly formatted. Now, please answer the question: {question}
Make sure to given anwers in short
Context metadata: {context_info}"""}
        ],
        max_tokens=200,
        temperature=TEMPERATURE)

        raw_response = response.choices[0].message.content.strip()
        print("Raw GPT Response:", raw_response)

        return raw_response

    except Exception as e:
        print(f"Error occurred in find_answer_gpt: {e}")
        return "Error occurred."

def test_highlight_pdf_text(pdf_path, highlight_text1):
    try:
        # Open the PDF file
        document = fitz.open(pdf_path)

        # Iterate through all pages
        for page_num in range(len(document)):
            page = document.load_page(page_num)  # PyMuPDF uses 0-based indexing
            text_instances = page.search_for(highlight_text1)

            if text_instances:
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.update()
                print(f"Highlighted '{highlight_text1}' on page {page_num + 1}")
            else:
                print(f"No matching text found on page {page_num + 1} for '{highlight_text1}'")

        # Save the modified PDF to a temporary in-memory buffer
        temp_file = io.BytesIO()
        document.save(temp_file)
        document.close()
        temp_file.seek(0)  # Ensure the stream is at the start

        return temp_file, None  # No error

    except Exception as e:
        print(f"Error in test_highlight_pdf_text: {e}")
        return None, e  # Return the error
def parse_gpt_response(response):
    try:
        # Directly parse the JSON response
        data = json.loads(response)
        return data
    except json.JSONDecodeError as je:
        print(f"JSONDecodeError: {je}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Streamlit interface for querying
print("Lag testing 5")
query = st.text_input("Enter your query:", key="query_input")
print("Lag testing 6")

if query:
    print("Lag testing 1")
    query_embedding = embed_chunks([query])[0]
    print("Lag testing 2")
    results = retrieve_documents(query_embedding)
    print("Lag testing 3")
    st.write("Going to retrieve documents")
    if results:
        with st.spinner("Assistant is typing..."):
            context, context_metadata = create_context_from_metadata(results)
            st.write("Context Metadata:")
            st.write(context_metadata)

            if context_metadata:
                answer = find_answer_gpt(query, context_metadata)
                st.write(f"Answer: {answer}")

                # Parse the JSON response from GPT
                data = parse_gpt_response(answer)

                if data:
                    try:
                        highlight_text = data.get("highlight", "")
                        filename = data.get("filename", "")
                        page_numbers = data.get("page_number", [])
                        #ans = data.get("ans", "")

                        # Display the direct answer
                        #st.write(f"Answer: {ans}")
                        directory = "/Users/sarthakgarg/Documents/Sarthak/"
                        pdf_path = os.path.join(directory, filename)


                        #pdf_path = "/Users/sarthakgarg/Documents/Sarthak/filename  # Replace with your PDF file path
                        highlight_text1 = highlight_text
                        output_path = get_next_file_path(base_directory, base_filename)

                        # Run the test function
                        highlighted_pdf, error = test_highlight_pdf_text(pdf_path, highlight_text1)
                        if highlighted_pdf:
                            print("Highlighting successful. You can now save the highlighted PDF.")
                            with open(output_path, "wb") as f:
                                f.write(highlighted_pdf.getbuffer())
                            print(f"Highlighted PDF saved as {output_path}")
                            st.success(f"PDF Highlight successful. File saved as {output_path.name}")
                        else:
                            print(f"Highlighting failed: {error}")
                            st.error(f"Highlighting failed: {error}")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        print(f"Exception: {e}")
                else:
                    st.error("Failed to parse the GPT response.")