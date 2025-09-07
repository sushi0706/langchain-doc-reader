# Multi-modal Doc Reader with RAG (LangChain + LLM)
A simple document Q&A system built with LangChain that understands both text and images, supporting RAG (Retrieval Augmented Generation).
You can load research papers, books, or notes, then ask questions in natural language — the system finds the relevant context and answers intelligently.

## Features
- Load and split documents into chunks (configurable size + overlap)
- Extracts and understands embedded images alongside text, allowing for Q&A that incorporates visual context.
- Embed chunks into a vector database for retrieval
- Query the document with natural language questions
- Conversation memory (follow-up questions supported)
