import os
import requests
import tempfile
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, HttpUrl

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import faiss

# Load environment variables
load_dotenv()
INDEX_PATH = "faiss_index"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Prompt
prompt_template = """You are a helpful assistant trained on insurance documents.
Use the following context to answer the user's question precisely, constructive and in one or two sentence. 
Not doing extra stuff with cost optimization.
Answer in a concise and complete manner.

Context:
{context}

Question:
{question}

Helpful Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# JSON Schema
class RAGRequestSchema(BaseModel):
    documents: HttpUrl
    questions: List[str]

# Load document from URL
def load_chunks_from_url(url: HttpUrl):
    response = requests.get(str(url))
    if response.status_code != 200:
        raise Exception("Failed to fetch file from URL")

    path = urlparse(str(url)).path.lower()
    suffix, LoaderClass = None, None
    if path.endswith(".pdf"):
        suffix, LoaderClass = ".pdf", PyPDFLoader
    elif path.endswith(".txt"):
        suffix, LoaderClass = ".txt", TextLoader
    elif path.endswith(".md"):
        suffix, LoaderClass = ".md", UnstructuredMarkdownLoader
    elif path.endswith(".html") or path.endswith(".htm"):
        suffix, LoaderClass = ".html", UnstructuredHTMLLoader
    else:
        raise Exception("Unsupported file type")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(response.content)
        file_path = tmp.name

    loader = LoaderClass(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

# Upload to FAISS
def upload_chunks_to_faiss(chunks):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(INDEX_PATH)

# Build vectorstore
def build_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(INDEX_PATH, embedding, allow_dangerous_deserialization=True)

# Get LLM from OpenRouter (DeepSeek)
def get_llm():
    return ChatOpenAI(
        model_name="deepseek/deepseek-r1-distill-llama-70b:free",
        openai_api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

# Build RAG chain
def build_rag_chain():
    retriever = build_vectorstore().as_retriever()
    llm = get_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False
    )

# Run ingestion
def run_pipeline(doc_url: HttpUrl):
    chunks = load_chunks_from_url(doc_url)
    upload_chunks_to_faiss(chunks)

# Batch Q&A
def get_answers_from_url_and_questions(doc_url: HttpUrl, questions: List[str]) -> List[str]:
    run_pipeline(doc_url)
    rag_chain = build_rag_chain()
    return [rag_chain.run(question) for question in questions]
