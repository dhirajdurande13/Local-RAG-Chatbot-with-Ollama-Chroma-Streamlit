import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from logger import logger

data=pd.read_csv('./data/realistic_restaurant_reviews.csv')
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
llm = Ollama(model="llama3.1:8b")
llm_fallback = Ollama(model="llama3.1:8b")
# creating embedings from the data in memory

def create_embeddings_from_df(df):
    if "Title" not in df.columns:
        raise ValueError("DataFrame must contain a 'Titel' column")

    
    memory_store = []
    chunk_size = 100 

    for _, row in df.iterrows():
        text = str(row["Title"]).strip()
        if not text:
            continue
        words = text.split()
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

        embeddings = model.encode(chunks, convert_to_numpy=True)

        for chunk, emb in zip(chunks, embeddings):
            memory_store.append({
                "text": chunk,
                "embedding": emb
            })

    return memory_store



def answer_question(memory_store, query, llm, top_k=3):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_emb = model.encode([query], convert_to_numpy=True)

    all_embeddings = np.array([m["embedding"] for m in memory_store])
    sims = cosine_similarity(query_emb, all_embeddings)[0]

    top_idx = sims.argsort()[-top_k:][::-1]
    retrieved_context = "\n".join([memory_store[i]["text"] for i in top_idx])

    QA_TEMPLATE = """
    You are an expert about answering questions about a pizza restaurant.
    Always give clear, helpful, and concise answers.

    Use the following pieces of retrieved context to answer the user’s question.
    If you don’t know the answer, try to get it from your knowledge and append this sentence: "I'm not sure about this".
    Keep answers in 2-3 lines.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=QA_TEMPLATE,
    )
    prompt_text = qa_prompt.format(context=retrieved_context, question=query)
    response = llm.invoke(prompt_text) 
    return response.content if hasattr(response, "content") else response



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


def run_query_without_db(query: str):
    memory_stored=create_embeddings_from_df(data)
    logger.info("Embedings created in memory")
    result=answer_question(memory_stored,query,llm)
    fall_res=process_with_fallback(result,query)
    return fall_res