



# Import necessary libraries
import os
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import gradio as gr
from transformers import pipeline


# Load environment variables
load_dotenv()
qdrant_api_key = os.getenv("QDRANT_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Prompt template for the LLM
# PROMPT_TEMPLATE = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know; don't try to make up an answer.
# Context: {context}
# Question: {question}

# Helpful answer:
# """
PROMPT_TEMPLATE = """
Context: {context}
Question: {question}

Helpful answer:
"""

# LLM configuration
LLM_CONFIG = {
    'max_new_tokens': 100,  # Reduce if memory usage is high
    'temperature': 0.1,
    'top_k': 5,
    'top_p': 0.9,
}

# Initialize Qdrant client
def initialize_qdrant():
    try:
        # Using REST API with no gRPC
        qdrant_client = QdrantClient(url="http://localhost", api_key=qdrant_api_key)
        hf_embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )
        
        # Set up Qdrant as vector store with REST API
        db = Qdrant(client=qdrant_client, embeddings=hf_embeddings, collection_name="Chatbox")
        retriever = db.as_retriever(search_kwargs={"k": 3})  # Top 3 results
        return retriever
    except Exception as e:
        print(f"Error initializing Qdrant: {str(e)}")
        return None

# Initialize the Hugging Face pipeline
#model="gpt2"
def initialize_llm():
    try:
        hf_pipeline = pipeline("text-generation", model="gpt2", **LLM_CONFIG)
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        return None

# Prompt setup for question answering
def prepare_prompt():
    
    return PromptTemplate(template=PROMPT_TEMPLATE, input_variables=['context', 'question'])

# Function to extract result text safely
def extract_result_text(response):
    result_text = response.get('result', 'I am not sure about that.')
    newline_position = result_text.find('\n\n')
    if newline_position != -1:
        result_text = result_text[:newline_position]
    return result_text

# Response generation function
def get_response(message, history):
    prompt = prepare_prompt()
    retriever = initialize_qdrant()
    llm = initialize_llm()
    
    if not retriever or not llm:
        yield "Error: Failed to initialize the retriever or LLM."
        return

    # Configure the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    response = qa({"query": message})
    result_text = extract_result_text(response)

    # Stream response with gradual typing effect
    for i in range(len(result_text)):
        time.sleep(0.05)
        yield result_text[:i+1]

# Gradio Chat Interface setup
chat = gr.ChatInterface(
    fn=get_response,
    chatbot=gr.Chatbot(label="My Chatbot", height=300),
    textbox=gr.Textbox(placeholder="Ask me something...", container=False, scale=3),
    title="Chatbot",
    description="This is a local data source LLM. It can make mistakes.",
    theme="soft",
    examples=["What is the main idea?", "Who are you?"],
    analytics_enabled=True
).queue()

if __name__ == '__main__':
    chat.launch()






