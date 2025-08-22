from src.helper import load_pdf, text_split, download_huggingface_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
 
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_huggingface_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'medical-bot-index'
index = pc.Index(name=index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)
