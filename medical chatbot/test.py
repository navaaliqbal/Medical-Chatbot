from langchain_community.vectorstores import FAISS
from embedder import get_embedding_model

embedding_model = get_embedding_model()
db = FAISS.load_local("vector stores/embeddings_faiss_index", embedding_model)
print("✅ FAISS index loaded!")
