import os
import hashlib
import uuid
import tqdm

from qdrant_client import models
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from . import config
from common.qdrant_client import QdrantClient

def get_pdf_files(directory):
    """
    Recursively finds all PDF files in a directory.
    """
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
            else:
                print(f"Skipping non-PDF file: {os.path.join(root, file)}")
    return pdf_files

def calculate_sha256(file_path):
    """
    Calculates the SHA256 hash of a file.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def check_hash_exists(qdrant_client, collection_name, file_hash):
    """
    Checks if a hash already exists in the Qdrant collection.
    """
    scroll_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.sha256",
                match=models.MatchValue(value=file_hash),
            )
        ]
    )
    
    response = qdrant_client.scroll_collection(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=1
    )
    
    if response and response[0]:
        return len(response[0]) > 0
    return False

def load_document_from_pdf(pdf_file_path):
    """
    loads documents from file path
    """
    document_loader = PyPDFLoader(pdf_file_path)
    doc = document_loader.load()
    print(f"📄 Loaded {len(doc)} pages")
    return doc

def get_embedding_function():
    model_name = config.EMBEDDING_MODEL
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def clean_text(text):
    """
    Cleans extracted PDF text by removing excessive whitespace and newlines.
    """
    import re
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove excessive newlines but keep paragraph breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    # Clean up spacing around newlines
    text = re.sub(r'\s*\n\s*', '\n', text)
    return text.strip()

def chunk_text(doc):
    """
    Splits doc into chunks of a specified size with overlap.
    """
    # Clean the text content of each document
    for document in doc:
        document.page_content = clean_text(document.page_content)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(doc)
    return chunks

def calculate_chunk_ids(chunks):
    """
    Creates unique chunk ids with filename:pageNumber:chunkIdx
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        filename = os.path.basename(chunk.metadata.get("source"))
        page = chunk.metadata.get("page")
        current_page_id = f"{filename}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
        chunk.metadata["filename"] = filename
        chunk.metadata["page"] = page

    return chunks

def ingest_pdf_files():
    """
    Ingests PDF files from the pdf directory.
    """
    pdf_dir = os.path.join(os.path.dirname(__file__), "pdf")
    pdf_files = get_pdf_files(pdf_dir)

    qdrant_client = QdrantClient()
    embedding_function = get_embedding_function()

    for file_path in tqdm.tqdm(pdf_files, desc="Processing PDFs"):
        file_hash = calculate_sha256(file_path)

        if check_hash_exists(qdrant_client, config.QDRANT_COLLECTION, file_hash):
            print(f"Skipping file '{os.path.basename(file_path)}' as it has already been ingested.")
            continue

        print(f"Processing new file: {os.path.basename(file_path)}")
        file = load_document_from_pdf(file_path)
        chunks = chunk_text(file)
        chunks_with_ids = calculate_chunk_ids(chunks)

        if not chunks_with_ids:
            continue

        chunk_texts = [chunk.page_content for chunk in chunks_with_ids]
        embeddings = embedding_function.embed_documents(chunk_texts)

        points = []
        for chunk, embedding in zip(chunks_with_ids, embeddings):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk.page_content,
                        "metadata": {
                            "sha256": file_hash,
                            "type": "pdf",
                            **chunk.metadata,
                        },
                    },
                )
            )
        
        if points:
            qdrant_client.upsert_points(
                collection_name=config.QDRANT_COLLECTION,
                points=points,
                wait=True,
            )
            print(f"Upserted {len(points)} points for file: {os.path.basename(file_path)}")


def ingest_documents():
    """
    Main ingestion function that processes PDF documents.
    """
    qdrant_client = QdrantClient()
    
    print("🗑️ Deleting existing collection...")
    qdrant_client.delete_collection(config.QDRANT_COLLECTION)

    print("🏗️ Creating new collection...")
    qdrant_client.create_collection(config.QDRANT_COLLECTION, config.VECTOR_SIZE)

    print("🔄 Ingesting PDF files...")
    ingest_pdf_files()
    print("✅ PDF document ingestion completed!")

if __name__ == "__main__":
    ingest_documents()
