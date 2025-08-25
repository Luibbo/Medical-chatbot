from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.llms import LlamaCpp
from src.prompt import prompt_template
import os
app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

embeddings = download_huggingface_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(name='medical-bot-index')

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
chain_type_kwargs = {'prompt': prompt}

llm = LlamaCpp(
    model_path='model/llama-2-13b-chat.Q4_K_M.gguf',
    n_ctx=2048,
    n_threads=0,
    temperature=0.8,
    max_tokens=512,
    verbose=False
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa.invoke({"query": input})
    print("Response: ", result['result'])
    return str(result['result'])


if __name__ == '__main__':
    app.run(debug=True)