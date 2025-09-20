import os
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
load_dotenv()  # make sure .env is loaded
DATA_DIR = "./data"


DB_FAISS_PATH = os.getenv("DB_FAISS_PATH", "vector_db")
def load_pdfs(data_dir):
    texts = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file)
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            texts.append({"filename": file, "content": text})
    return texts

def build_vectorstore(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([t["content"] for t in texts])#list comprehension to extract content

    # Save FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

 

    faiss.write_index(index, f"{os.getenv('DB_FAISS_PATH')}.index")

    # Save metadata
    with open(f"{DB_FAISS_PATH}_meta.pkl", "wb") as f:
        pickle.dump(texts, f)

    print(f"✅ Vector database saved to {DB_FAISS_PATH}.index")

if __name__ == "__main__":
    texts = load_pdfs(DATA_DIR)
    build_vectorstore(texts)
