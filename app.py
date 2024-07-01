import os
from flask import Flask, render_template, request
from src.utils import download_hugging_face_embeddings, load_pdf, text_split
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

extracted_data = load_pdf("./notebook/data/")

text_chunks = text_split(extracted_data)

embeddings = download_hugging_face_embeddings()

#Loading the index
vectorstore = Chroma.from_documents(text_chunks,embeddings)

llm=CTransformers(model="./model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

@app.route("/")
def index():
    return render_template('chatbot.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = retrieval_chain.invoke({"input": msg})
    return str(response['answer'])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
