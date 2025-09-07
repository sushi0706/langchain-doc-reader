from langchain_core.messages import HumanMessage

def build_message(query, docs):
    content = [{"type": "text", "text": f"User Question: {query}\n\n"}]

    for doc in docs:
        content.append({
            "type": "text",
            "text": f"Document Content: {doc.page_content}\n"
        })

        if "image_base64" in doc.metadata:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{doc.metadata['image_base64']}"}
            })

    return HumanMessage(content=content)