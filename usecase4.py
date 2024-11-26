import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber  


model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embeddings(texts):
    return np.array(model.encode(texts))


def chunk_text(text, chunk_size=350, overlap=100):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks

# Function to extract text from PDF files
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # Avoid None if text extraction fails
    return text.strip()


st.title("RAG Framework to Retrieve Relevant Legal Documents")

uploaded_files = st.file_uploader("Upload Legal Documents", accept_multiple_files=True, type=['pdf', 'txt'])
query = st.text_input("Enter your legal query")


docs = []
chunks = []
if st.button('Submit'):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.type == ['pdf','txt']:
               
                doc_text = extract_text_from_pdf(uploaded_file)
            else:
                # Handle text files
                try:
                    doc_text = uploaded_file.read().decode("utf-8", errors='ignore')
                except UnicodeDecodeError:
                    doc_text = uploaded_file.read().decode('latin1')
            
            if doc_text: 
                docs.append(doc_text)

                
                doc_chunks = chunk_text(doc_text, chunk_size=400, overlap=100)
                chunks.extend(doc_chunks)

    if query and docs:
        
        chunk_embeddings = generate_embeddings(chunks)
        
        # Ensure chunk_embeddings is 2D
        if len(chunk_embeddings.shape) == 1:
            chunk_embeddings = chunk_embeddings.reshape(1, -1)

        # Build FAISS index
        d = chunk_embeddings.shape[1]  
        index = faiss.IndexFlatL2(d)  # Create a FAISS index
        index.add(chunk_embeddings)     # Adding chunk embeddings to the index

        # Generate embedding for the query
        query_embedding = generate_embeddings([query])
        
        # Ensure query_embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Limit top_k to the number of unique chunks
        top_k = min(len(chunks), 5)

        # Perform FAISS search
        D, I = index.search(query_embedding, k=top_k)  # Get top 5 closest chunks

        # Display results
        st.write("Top retrieved chunks:")
        for index in I[0]:  # Use the indices retrieved from FAISS
            st.write(f"Chunk {index + 1}:")
            st.write(chunks[index])  # Display the relevant chunk

            # Placeholder for summary/response generation
            st.write("Generated Summary/Response:", f"Summary of Chunk {index + 1}.")
