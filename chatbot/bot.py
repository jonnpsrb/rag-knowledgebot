from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from . import config
from common.qdrant_client import QdrantClient


def get_embedding_function():
    model_name = config.EMBEDDING_MODEL
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def get_vector_store():
    qdrant_client = QdrantClient()
    return QdrantVectorStore(
        client=qdrant_client.get_client(),
        collection_name=config.QDRANT_COLLECTION,
        embedding=get_embedding_function(),
        content_payload_key="text",
        metadata_payload_key="metadata",
    )

def get_pdf_retriever():
    from qdrant_client.http import models
    vector_store = get_vector_store()
    return vector_store.as_retriever(
        search_kwargs={
            "k": config.RETRIEVAL_TOP_K,
            "score_threshold": config.RETRIEVAL_SCORE_THRESHOLD,
            "filter": models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.type",
                        match=models.MatchValue(value="pdf")
                    )
                ]
            )
        }
    )

def get_rag_chain():
    pdf_retriever = get_pdf_retriever()

    llm = ChatGroq(api_key=config.GROQ_API_KEY, model=config.LLM_MODEL_NAME)

    prompt = ChatPromptTemplate.from_messages([
        ("system", config.BOT_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}\n\nContext provided by RAG system:\n{context}"),
    ])

    def format_docs_with_source(docs, source_type):
        if not docs:
            return ""
        formatted = f"\n=== {source_type.upper()} SOURCES ===\n"
        for doc in docs:
            source_info = ""
            if doc.metadata.get("type") == "pdf":
                source_info = f" (Source: {doc.metadata.get('filename', 'Unknown')}, Page {doc.metadata.get('page', 'N/A')})"
            formatted += f"{doc.page_content}{source_info}\n\n"
        return formatted

    def get_context(inputs):
        question = inputs.get("question", inputs.get("input", ""))

        # Retrieve pdf source
        pdf_docs = pdf_retriever.invoke(question)
        
        # Format with source identification
        pdf_context = format_docs_with_source(pdf_docs, "PDF Documentation")
            
        return pdf_context
    
    rag_chain = (
        {
            "context": get_context,
            "question": lambda x: x.get("question", x.get("input", "")),
            "history": lambda x: x.get("history", []),
        }
        | prompt
        | llm
    )

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return chain_with_history
