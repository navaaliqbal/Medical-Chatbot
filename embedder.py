from preprocessing import create_segments
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def get_embedding_model(model_name= "sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embedding_model

text_chunks = create_segments()

DB_PATH = "vector stores/embeddings_faiss_index"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

vectorstore = FAISS.from_documents(text_chunks, get_embedding_model())
vectorstore.save_local(DB_PATH)


