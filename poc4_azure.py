import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.ai.textanalytics import TextAnalyticsClient
from openai import AzureOpenAI
import os
import tempfile

# Azure service configurations
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_key = os.getenv("AZURE_OPENAI_KEY")
form_recognizer_endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
form_recognizer_key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")

# Initialize Azure services
search_client = SearchClient(search_endpoint, search_index_name, AzureKeyCredential(search_key))
openai_client = AzureOpenAI(api_key=openai_key, api_version="2023-05-15", azure_endpoint=openai_endpoint)
document_analysis_client = DocumentAnalysisClient(form_recognizer_endpoint, AzureKeyCredential(form_recognizer_key))

def process_document(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name

    with open(temp_file_path, "rb") as f:
        poller = document_analysis_client.begin_analyze_document("prebuilt-document", f)
    result = poller.result()

    # Extract text content from the analyzed document
    document_text = " ".join([page.content for page in result.pages])

    # Index the document in Azure Cognitive Search
    search_client.upload_documents([{
        "id": file.name,
        "content": document_text
    }])

    os.unlink(temp_file_path)
    return document_text

def retrieve_documents(query):
    results = search_client.search(query)
    return [result['content'] for result in results]

def generate_response(query, context):
    prompt = f"Based on the following legal context:\n\n{context}\n\nProvide legal advice for the query: {query}"
    response = openai_client.chat.completions.create(
        model="gpt-4",  # or your deployed model name
        messages=[{"role": "system", "content": "You are a legal assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

st.title("Legal Advice RAG System")

uploaded_file = st.file_uploader("Upload a legal document", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    with st.spinner("Processing document..."):
        document_text = process_document(uploaded_file)
    st.success(f"Document '{uploaded_file.name}' processed and indexed successfully!")

query = st.text_input("Enter your legal question:")

if st.button("Get Advice"):
    with st.spinner("Retrieving relevant documents..."):
        relevant_docs = retrieve_documents(query)
        context = "\n".join(relevant_docs)

    with st.spinner("Generating legal advice..."):
        advice = generate_response(query, context)

    st.subheader("Legal Advice:")
    st.write(advice)

    st.subheader("Relevant Document Excerpts:")
    for doc in relevant_docs:
        st.write(doc[:200] + "...")  # Display first 200 characters of each document