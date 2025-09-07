from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from core.loader import load_pdf
from core.splitter import split_documents
from core.embeddings import get_embeddings
from core.vector_store import get_vector_store
from utils.print_helper import print_response
from utils.msg_formatter import build_message
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
    print("Vector store created and persisted")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectordb.as_retriever()

    while True:
        query = input("\nYour Question (or 'exit' to quit): ")

        if query.lower() == 'exit':
            print("Exiting the program.")
            break

        docs = retriever.get_relevant_documents(query)
        message = build_message(query, docs)

        history = memory.chat_memory.messages if memory.chat_memory.messages else []
        response = llm.invoke(history + [message])

        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response.content)
        print_response(response)

if __name__ == "__main__":
    main()