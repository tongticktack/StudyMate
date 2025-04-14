from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import json

def embedding_title():
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    with open("./document.json", 'r') as f:
        document_json = json.load(f)
        
    documents = list(document_json.keys())

    faiss_index = FAISS.from_texts(documents, embedding_model)
    faiss_index.save_local("./DB/faiss_index")

if __name__=='__main__':
    embedding_title()