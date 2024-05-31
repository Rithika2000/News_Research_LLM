import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

file_path = "vector_index.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    try:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        embeddings = HuggingFaceEmbeddings()
        vector_index = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        with open(file_path, "wb") as f:
            pickle.dump(vector_index, f)
    except Exception as e:
        main_placeholder.text(f"Error processing URLs: {e}")

query = main_placeholder.text_input("Question: ")

if query:
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vector_index = pickle.load(f)
                embeddings = HuggingFaceEmbeddings()
                query_embeddings = embeddings([query])
                D, I = vector_index.search(query_embeddings, k=5)
                st.header("Top 5 Similar Documents")
                for i in range(5):
                    st.write(docs[I[i]])
    except Exception as e:
        st.write(f"Error processing query: {e}")
