from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Step 1: Load PDF
data = PyPDFLoader(r"document loaders/deeplearning.pdf")
docs = data.load()
print(f"✅ Loaded {len(docs)} pages")

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"✅ Split into {len(chunks)} chunks")

# Step 3: Create embeddings
embeddings = MistralAIEmbeddings(model="mistral-embed")  # Fixed model name

# Step 4: Store in vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
print("✅ Stored in Chroma vector database!")