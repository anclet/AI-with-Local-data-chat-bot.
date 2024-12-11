# ingest_documents.py

# ingest_documents.py
import os
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from dotenv import load_dotenv

##############################################

pdf_files = 'data'
loader = PyPDFDirectoryLoader(f'data')
documents = loader.load()
print(loader)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=765,
    chunk_overlap=50,
)

texts = text_splitter.split_documents(documents)

# Loading Embedding Model
model_name = "sentence-transformers/all-MiniLM-L12-v2"
model_kwargs = {"device": "cpu"}  # Use CPU (change to "cuda" for GPU if available)
encode_kwargs = {"normalize_embeddings": False}  # Whether to normalize embeddings

###############################################

hf = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

print('Embeddings Loader____')

# Qdrant connection details
url = "http://localhost"
collection_name = "Chatbox"

# Recreating the collection with the correct embedding dimensions (384)
Qdrant = Qdrant.from_documents(
    texts,
    hf,
    url=url,
    prefer_grpc=True,
    collection_name=collection_name,
    force_recreate=True  # Force recreation of the collection with the correct dimensions
)

print("Database index created_____")






# import os
# from langchain_community.vectorstores import Qdrant
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from qdrant_client import QdrantClient
# from dotenv import load_dotenv

# ##############################################

# pdf_files=f'data'
# loader=PyPDFDirectoryLoader(f'data')
# documents=loader.load()
# print(loader)


# text_splitter=RecursiveCharacterTextSplitter(
#     chunk_size = 765,
#     chunk_overlap = 50,
# )

# texts=text_splitter.split_documents(documents)

# #loading Embending Model

# #model_name = "jinaai/jina-embeddings-v2-base-en"
# # model_kwargs = {"device": "cpu"}
# # encode_kwargs = {"normalize_embeddings": False}
# #model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# model_kwargs = {"device": "cpu"}  # Use CPU (change to "cuda" for GPU if available)
# encode_kwargs = {"normalize_embeddings": False}  # Whether to normalize embeddings

# ###############################################

# hf =HuggingFaceEmbeddings(
#     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
# )

# print('Embeddings Loader____')

# #url="http://host.docker.internal"
# url="http://localhost"
# collection_name = "Chatbox"
# Qdrant = Qdrant.from_documents(
#     texts,
#     hf,
#     url=url,
#     prefer_grpc= True,
#     collection_name=collection_name
# )

# print("database index created_____")
#####################################################################################################

# # Load environment variables
# load_dotenv()
# qdrant_api_key = os.getenv("QDRANT_API_KEY")

# # Initialize Qdrant client
# client = QdrantClient(url="http://localhost", port=6333)
# collection_name = "Chatbox"

# # Load and split documents
# pdf_dir = 'data'
# loader = PyPDFDirectoryLoader(pdf_dir)
# documents = loader.load()
# print("Loaded documents:", len(documents))

# # Split documents into chunks for efficient embedding
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=768, chunk_overlap=50)
# texts = text_splitter.split_documents(documents)

# # Initialize embedding model
# hf_embeddings = HuggingFaceEmbeddings(
#     model_name="jinaai/jina-embeddings-v2-base-en",
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": False}
# )

# # # Store documents in Qdrant with embeddings
# # Qdrant.from_documents(
# #     texts,
# #     hf_embeddings,
# #     url="http://localhost:6333",  # Ensure using the correct port 6333 for HTTP
# #     prefer_grpc=False,  # Set to False to avoid using gRPC (since it's on a different port)
# #     collection_name=collection_name
# # )

# print("Database index created.")
