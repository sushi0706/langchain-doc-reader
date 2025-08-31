from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(pages, chunk_size=1000, chunk_overlap=100):
    """
    Split documents into smaller chunks
    """
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = r_splitter.split_documents(pages)
    return splits