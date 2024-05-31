import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token="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")


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
        vector_index_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        with open(file_path, "wb") as f:
            pickle.dump(vector_index_openai, f)
    except Exception as e:
        main_placeholder.text(f"Error processing URLs: {e}")

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_Index = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,
                                 retriever=vector_Index.as_retriever)
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])
            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
                
                
                
                
                
                