import streamlit as st
import requests

# Define the base URL for your Flask API
base_url = "http://localhost:5001"

# Set up Streamlit app title
st.title("RAG System GAIL")

# Sidebar for navigation
option = st.sidebar.selectbox(
    "Choose an action",
    ("Upload File", "Query", "View Filenames", "Delete File", "Clear History", "View History")
)

# Upload File
if option == "Upload File":
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file to upload", type="pdf")
    if uploaded_file is not None:
        files = {'file': uploaded_file}
        response = requests.post(f"{base_url}/upload", files=files)
        if response.status_code == 200:
            st.success("File uploaded and processed successfully")
        else:
            st.error(f"Failed to upload file: {response.json().get('error')}")

# Query PDFs
elif option == "Query":
    st.header("Query PDFs")
    user_query = st.text_input("Enter your query:")
    user_id = st.text_input("Enter user ID (optional):", value="default")
    filenames = st.text_input("Enter filenames (optional, comma-separated):")
    
    if st.button("Submit Query"):
        data = {
            "user_id": user_id,
            "query": user_query,
            "filenames": filenames.split(",") if filenames else None
        }
        #st.write(f"Data sent to API: {data}")
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{base_url}/query", json=data)
        if response.status_code == 200:
            result = response.json()
            
            # Display the answer prominently with larger text
            st.subheader("Answer")
            st.markdown(f"<div style='font-size:24px; color:white;'>{result.get('ans')}</div>", unsafe_allow_html=True)

            # Display highlights in a structured format with larger text
            st.subheader("Highlights")
            for highlight in result.get("highlight", []):
                st.markdown(f"**Filename**: `{highlight['filename']}`")
                st.markdown(f"**Page Number**: `{highlight['page_number']}`")
                st.markdown(f"<div style='font-size:20px; color:green;'>\"{highlight['text']}\"</div>", unsafe_allow_html=True)
                st.markdown("---")  # Horizontal line for separation between highlights
        else:
            st.error(f"Query failed: {response.json().get('error')}")

# View Filenames
elif option == "View Filenames":
    st.header("View Filenames")
    response = requests.get(f"{base_url}/filenames")
    if response.status_code == 200:
        filenames = response.json().get("filenames", [])
        
        if filenames:
            st.subheader("Uploaded Filenames")
            st.markdown("<ul style='list-style-type:disc;'>", unsafe_allow_html=True)
            for index, filename in enumerate(filenames):
                st.markdown(f"<li style='font-size:18px; color:;'>{index+1}. {filename}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
        else:
            st.info("No files have been uploaded yet.")
    else:
        st.error(f"Failed to retrieve filenames: {response.json().get('error')}")


# Delete File
elif option == "Delete File":
    st.header("Delete a File")
    filename_to_delete = st.text_input("Enter the filename to delete:")
    if st.button("Delete"):
        data = {"filename": filename_to_delete}
        response = requests.post(f"{base_url}/delete", json=data)
        if response.status_code == 200:
            st.success("File deleted successfully")
        else:
            st.error(f"Failed to delete file: {response.json().get('error')}")

# Clear History
elif option == "Clear History":
    st.header("Clear History")
    if st.button("Clear"):
        response = requests.post(f"{base_url}/clear")
        if response.status_code == 200:
            st.success("Chat history cleared successfully")
        else:
            st.error(f"Failed to clear chat history: {response.json().get('error')}")

# View History
elif option == "View History":
    st.header("View History")
    user_id = st.text_input("Enter user ID to view history:", value="default")
    if st.button("View"):
        response = requests.get(f"{base_url}/history", params={"user_id": user_id})
        if response.status_code == 200:
            history = response.json()
            st.write(history)
        else:
            st.error(f"Failed to retrieve history: {response.json().get('error')}")
