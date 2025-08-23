# 🧠 Local RAG Chatbot with Ollama + Chroma + Streamlit

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** running fully **locally**, using:

- [Ollama](https://ollama.ai) → to run LLMs like `llama3`.  
- [LangChain](https://www.langchain.com) → orchestration for RAG.  
- [Chroma](https://www.trychroma.com) → local vector database for document embeddings.  
- [Streamlit](https://streamlit.io) → simple interactive UI.  

---

## 🚀 Features
- use documents (csv).  
- Store embeddings in **Chroma** (persisted locally).  
- Run queries against your knowledge base.  
- Responses are generated using **Ollama LLMs**, fully offline.  
- Caches embeddings so repeated runs are faster.  

---

## 📂 Project Structure

Learning Model Locally/
│── main.py # Streamlit app entrypoint
│── db_utils.py # Database utils (Chroma vector store)
│── qa_utils.py # Query + RAG pipeline
│── requirements.txt # Python dependencies
│── README.md # Project documentation
└── data/ # Folder for storing uploaded files