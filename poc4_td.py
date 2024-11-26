import os
from pypdf import PdfReader
import google.generativeai as genai
import faiss
import numpy as np
from dotenv import load_dotenv
import streamlit as st
from sentence_transformers import SentenceTransformer


load_dotenv()


# Gemini API setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Document processing functions
def extract_text_from_pdf(pdf_file):
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text


def preprocess_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)
            documents.append({"content": text, "source": uploaded_file})
    return documents


# def embed_documents(documents):
#     embeddings = []
#     embed_model = SentenceTransformer('all-MiniLM-L6-v2')
#     for doc in documents:
#         embedding = embed_model.encode(doc["content"])
#         embeddings.append(embedding)
#     return embeddings

# Embedding and vector database functions
def embed_documents(documents):
    # model = genai.GenerativeModel('embedding-001')
    # model = genai.get_model('models/text-embedding-004')
    # embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    for doc in documents:
        # embedding_input = doc["content"]
        # embedding_response = model.embed(text = doc["content"])
        # embedding = embedding_response['embedding']
        # embedding = embedding_model.get_embeddings([embedding_input])
        # embeddings.append(embedding.embeddings[0])
        # print(type(model))
        # print(doc)
        embedding = model.encode(doc["content"], convert_to_tensor=False)
        embeddings.append(embedding)
    return np.array(embeddings)


def create_vector_db(embeddings):
    # dimension = len(embeddings[1])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array(embeddings).astype('float32')
    index.add(vectors)
    return index

def retrieve_documents(query, index, documents, top_k=3):
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')   
    # query_embedding = embed_model.encode(query)
    # model = genai.get_model('models/text-embedding-004')
    # embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    
    # query_embedding = model.embed_text(query)
    # query_embedding = embedding_model.get_embeddings([query])
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    query_embedding = np.array([query_embedding], dtype= 'float32')
    
    # # Extract the embedding
    # query_embedding = query_embedding.embeddings[0]
    
     # Perform a similarity search using FAISS
    _, I = index.search(query_embedding, top_k)  # No need to cast to float32 again; it's already done

    # Retrieve the top_k most similar documents
    retrieved_docs = [documents[i] for i in I[0]]
    
    return retrieved_docs
    # _, I = index.search(np.array([query_embedding]).astype('float32'), top_k)
    # retrieved_docs = [documents[i] for i in I[0]]
    # return retrieved_docs

def generate_response(query, retrieved_docs):
    model = genai.GenerativeModel('gemini-pro')
    
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])
    prompt = f"""
    Context: {context}
    
    Human: {query}
    
    Assistant: Based on the provided legal documents, here's my analysis and advice:
    """
    
    response = model.generate_content(prompt)
    return response.text

# Streamlit app
def main():
    st.title("Legal Document Analysis with RAG")

    # File uploader
    uploaded_files = st.file_uploader("Upload legal documents (PDF)", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Processing documents..."):
            documents = preprocess_documents(uploaded_files)
            if not documents:
                st.error("No valid documents were uploaded. Please upload PDF files.")
                return

            embeddings = embed_documents(documents)
            if embeddings.size ==0:
                st.error("Failed to create document embeddings. Please try again.")
                return

            vector_db = create_vector_db(embeddings)
            if vector_db is None:
                st.error("Failed to create vector database. Please try again.")
                return

        st.success(f"Processed {len(documents)} documents successfully!")

        # Query input
        query = st.text_input("Enter your legal query:")

        if query:
            with st.spinner("Analyzing..."):
                retrieved_docs = retrieve_documents(query, vector_db, documents)
                if not retrieved_docs:
                    st.warning("No relevant documents found. Try rephrasing your query.")
                    return

                response = generate_response(query, retrieved_docs)
                st.subheader("Analysis:")
                st.write(response)

                st.subheader("Retrieved Documents:")
                for doc in retrieved_docs:
                    st.write(f"- {doc['source']}")

if __name__ == "__main__":
    main()