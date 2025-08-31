import yaml
from langchain_community.vectorstores import Chroma

def get_vector_store(splits, embeddings):
    """
    Create a Chroma vector store from documents and embeddings
    """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    persist_directory = config.get("persist_dir")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectordb