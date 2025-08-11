from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from embedder import get_embedding_model
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

if "MISTRAL_API_KEY" not in os.environ:
    raise EnvironmentError("MISTRAL_API_KEY not set in environment variables")

llm = ChatMistralAI(
    model="mistral-small-latest", 
    temperature=0,                 
    max_retries=2
)
DB_PATH = "vector stores/embeddings_faiss_index"

embedding_model = get_embedding_model()
print("loading faiss index")
db= FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)

print("Initializing LLM")


PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}
"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
retriever = db.as_retriever(search_kwargs={'k':3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever = retriever,
    chain_type="stuff",
    chain_type_kwargs={'prompt': prompt},
    return_source_documents=True
)

if __name__ == "__main__":
    print("Medical Chatbot with RetrievalQA Chain")
    while True:
        query = input("\nAsk your question (or 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break

        result = qa_chain.invoke({"query": query})

        print("\nAnswer:\n", result["result"])

        print("\ Sources:\n")
        for doc in result["source_documents"]:
            print("-", doc.metadata.get("source", "Unknown"))





