from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import os
from logger import logger


llm = Ollama(model="llama3.1:8b")
llm_fallback = Ollama(model="llama3.1:8b")
# creating retriver
def get_retriever():
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    # check is the data loaded or not
    persist_dir = "./chroma_db_sent"
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        logger.error("ChromaDB not found. Please run db_utils.py to load data first.")
        raise FileNotFoundError("ChromaDB not found. Run db_utils.py to load embeddings before retrieving.")
    
    vectorstore = Chroma(
        collection_name="collection",   
        persist_directory="./chroma_db_sent",
        embedding_function=embedding_function
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",   
        search_kwargs={"k": 3}    
    )
    # logger.info("Retriever successfully created from ChromaDB.")
    return retriever

retriever=get_retriever()


def get_qa_chain(llm, retriever):
    QA_TEMPLATE = """
    You are an expert about answering questions about a pizza restaurant.
    Always give clear, helpful, and concise answers.

    Use the following pieces of retrieved context to answer the user’s question.
    If you don’t know the answer, try to get it from your knowledge and append this sentence, "I'm not sure about this". Don’t make things up.
    And make sure answer must be in 2 to 3 lines.

    Context:
    {context}

    Question: {question}

    Answer:"""

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=QA_TEMPLATE,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, 
        chain_type="stuff",  # For converting context into prompt
        chain_type_kwargs={"prompt": qa_prompt}
    )

    return qa_chain

# this will work if the question is outsider or which is not releated to the document
def process_with_fallback(result: str, query: str) -> str:
    irrelevant_markers = [
        "doesn't seem to be about",
        "this conversation doesn't seem",
        "i'd be happy to try and help",
        "not related",
        "i'm not sure about this",
        "this question is not relevant",
        "out of context"
    ]
    if any(marker.lower() in result.lower() for marker in irrelevant_markers):
        # print("Irrelevant result detected, switching to fallback LLM...")
        prompt = f"""
            Please answer the following question in only 2–3 concise lines:

            Question: {query}
            """
        fallback_answer = llm_fallback.invoke(prompt)
        return fallback_answer
    
    return result

def run_query(query: str):
    qa_chain=get_qa_chain(llm,retriever)
    result=qa_chain.run({"query": query})
    fall_res=process_with_fallback(result,query)
    return fall_res