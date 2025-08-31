from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from core.loader import load_pdf
from core.splitter import split_documents
from core.embeddings import get_embeddings
from core.vector_store import get_vector_store
from utils.print_helper import print_response
import yaml
from dotenv import load_dotenv

load_dotenv()

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    doc_path = config.get("doc_path")

    pages = load_pdf(doc_path)
    splits = split_documents(pages)
    embeddings = get_embeddings()
    vectordb = get_vector_store(splits, embeddings)
    print("Vector store created and persisted.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectordb.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )

    while True:
        query = input("\nYour Question (or 'exit' to quit): ")

        if query.lower() == 'exit':
            print("Exiting the program.")
            break

        response = qa_chain.invoke({"question": query})
        print_response(response)

if __name__ == "__main__":
    main()