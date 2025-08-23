import pandas as pd
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from logger import logger
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from langchain.embeddings import HuggingFaceEmbeddings
data = pd.read_csv('./data/realistic_restaurant_reviews.csv')

def store_in_chroma(df):
    if "Review" not in df.columns:
        raise ValueError("DataFrame must contain a 'Review' column")
    
    texts = df["Review"].dropna().astype(str).tolist()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    documents = text_splitter.create_documents(texts)
    print('documents created')

    # Make sure nomic-embed-text is available in Ollama
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print('embeddings ready',embeddings)

    persist_dir = "./chroma_db"
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma(
        collection_name="collection",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    existing_docs = vectorstore.get(include=["documents"])
    existing_texts = set(existing_docs["documents"]) if existing_docs else set()
    
    new_texts = [doc.page_content for doc in documents if doc.page_content not in existing_texts]
    print("New text found:", len(new_texts))

    if new_texts:
        print("Adding in vectorstore...")
        vectorstore.add_texts(new_texts)
        print('Added in vectorstore')
        vectorstore.persist()
        
    else:
        print("No new data to add, everything already exists in ChromaDB")

    updated_docs = vectorstore.get(include=["documents"])
    total_count = len(updated_docs["documents"]) if updated_docs else 0
    return vectorstore


persist_dir = "./chroma_db"
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    logger.info("Data already exists in ChromaDB. Skipping embedding creation.")
else:
    print('called')
    vectorstore = store_in_chroma(data)
    logger.info(f"Data loaded into ChromaDB: {vectorstore} chunks")
    print(f"Inserted {vectorstore}  ChromaDB")

print("Chroma db data insertion completed")
