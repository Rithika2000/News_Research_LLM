# News_Research_LLM

![Newsbot](https://github.com/Rithika2000/News_Research_LLM/assets/57192807/2171690f-b9f1-43db-aeaf-3e6724a0e5e5)

# Features

1.Load URLs or upload text files containing URLs to fetch article content.
2.Process article content through LangChain's UnstructuredURL Loader
3.Construct an embedding vector using HuggingFace embeddings and leverage FAISS, a powerful similarity search library, to enable swift and 4.Effective retrieval of relevant information
5.Interact with the LLM's (Chatgpt) by inputting queries and receiving answers along with source URLs.

# Steps to run the app

1.Run the Streamlit app by executing:

streamlit run Retail.url.py

2.The web app will open in your browser.

3.On the sidebar, you can input URLs directly.

4.Initiate the data loading and processing by clicking "Process URLs."

5.Observe the system as it performs text splitting, generates embedding vectors, and efficiently indexes them using FAISS.

6.The embeddings will be stored and indexed using FAISS, enhancing retrieval speed.

7.The FAISS index will be saved in a local file path in pickle format for future use.

8.One can now ask a question and get the answer based on those news articles

9.Below are the news article url's used 

https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html

https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html

https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html

# Project Structure

Retail.url.py: The main Streamlit application script.

vector_index.pkl: A pickle file to store the FAISS index.



