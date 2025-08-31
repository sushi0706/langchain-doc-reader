from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings(model_name="all-MiniLM-L6-v2"):
    """
    Get HuggingFace embeddings model
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings