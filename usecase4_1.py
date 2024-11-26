import google.generativeai as genai
import os
import streamlit as st
import faiss
import numpy as np
import pdfplumber
from IPython import display

genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))

# for m in genai.list_models():
#   if 'embedContent' in m.supported_generation_methods:
#     print(m.name)

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # Avoid None if text extraction fails
    return text

def chunk_document(document_text, chunk_size=300, overlap=50):

  words = document_text.split()  # Split text into words
  chunks = []
  
  for i in range(0, len(words), chunk_size - overlap):
      chunk = " ".join(words[i:i + chunk_size])
      chunks.append(chunk)
  
  return chunks

def generate_embedding(text):
  model = 'models/embedding-001'
  embedding = genai.embed_content(
      model=model,
      content=text,
      task_type="retrieval_document"
  )
  return embedding

def build_faiss_index(document_text):
    chunks = chunk_document(document_text)
    embeddings = []
    for chunk in chunks:
        embedding_response = generate_embedding(chunk)  # it output is in dict format
        # print(type(embedding_response))
        embedding = embedding_response.get('embedding') 
        if embedding is not None:
            embeddings.append(embedding)
        else:
            st.warning(f"Embedding not found for chunk: {chunk}")

    embedding_matrix = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # L2 distance
    index.add(embedding_matrix)  # Add embeddings to the index
    return index, chunks

def search_similar_documents(query, index , chunks):
  query_embedding = generate_embedding(query)
  query_embedding = np.array(query_embedding.get('embedding')).astype('float32').reshape(1, -1)
  distances, indices = index.search(query_embedding, 5)  # Search for top 5 nearest chunks
  return [chunks[i] for i in indices[0]]

def legal_response(query,similar_chunks):
  model = genai.GenerativeModel("gemini-1.5-flash")
  response = model.generate_content(f"I know you are not the Legal adviser.Please Go through the {similar_chunks } and try to understand the concept.Generate a rough Legal Adive for :{query}")
  return response.text

st.title("RAG Framework to Retrieve Relevant Legal Documents")

uploaded_files = st.file_uploader("Upload Legal Documents", accept_multiple_files=True, type=['pdf', 'txt'])
query = st.text_input("Enter your legal query")

docs = []
chunks = []
if st.button('Submit'):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.type in ['application/pdf', 'text/plain']:  # Check for correct MIME types
                if uploaded_file.type == 'application/pdf':
                    doc_text = extract_text_from_pdf(uploaded_file)
                else:
                    # Handle text files
                    doc_text = uploaded_file.read().decode("utf-8", errors='ignore')
                
                if doc_text: 
                    docs.append(doc_text)
                    doc_chunks = chunk_document(doc_text)
                    chunks.extend(doc_chunks)

    if query and docs:
        index,_ = build_faiss_index(" ".join(docs))
        similar_chunks = search_similar_documents(query, index, chunks) # doing the similar search
        print(type(similar_chunks))
        # for i, chunk in enumerate(similar_chunks):
        #     st.write(f"Chunk {i + 1}:")
        #     st.write(chunk)
        st.text_area("Retrived response" , similar_chunks[:4000], height= 200)
        # st.write(similar_chunks)
        generated_reponse  = legal_response(query, similar_chunks)
        st.subheader("Legal Advice")
        st.write(generated_reponse)        

