import pickle
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
import os

DB_FAISS_PATH = os.getenv("DB_FAISS_PATH", "vector_db")
# Load FAISS index and metadata
index = faiss.read_index(f"{DB_FAISS_PATH}.index")
with open(f"{DB_FAISS_PATH}_meta.pkl", "rb") as f:
    texts = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_context(query, top_k=5):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, top_k)
    results = [texts[i]["content"] for i in I[0]]
    return "\n\n".join(results)

# Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

# Example query
user_query = "What are the best practices for raising a pet iguana?"

context = get_context(user_query)
while True:

    Content=input("Enter your question here: ")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that answers questions based on provided context. The data is of the resume's only so don't make up answers if the context doesn't have the information. If you don't know the answer, just say that you don't know. Do not try to make up an answer.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {Content}",
            },
        ],
        model="llama-3.3-70b-versatile",
    )
    
    print("Assistant:", chat_completion.choices[0].message.content)
