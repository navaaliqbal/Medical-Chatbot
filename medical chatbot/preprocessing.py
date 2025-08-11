from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_segments(file_path="data/Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf"):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    segments = splitter.split_documents(docs)
    return segments



