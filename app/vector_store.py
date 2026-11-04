
import chromadb
import pymupdf
import hashlib
import time
import json

from pathlib import Path
from sentence_transformers import SentenceTransformer

from app.pdf_loader import extract_text
from app.chunker import create_chunks


# --------------------------------
# CHROMADB INITIALIZATION
# --------------------------------

project_folder = Path(__file__).resolve().parent.parent

client = chromadb.PersistentClient(
    path=str(project_folder / "chroma_db")
)


# --------------------------------
# EMBEDDING MODEL
# --------------------------------

embedding_model = None


def get_embedding_model():

    global embedding_model

    if embedding_model is None:

        print("Loading embedding model...")

        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    return embedding_model


# --------------------------------
# COLLECTION MANAGEMENT
# --------------------------------

def collection_exists(collection_name):

    try:

        client.get_collection(name=collection_name)

        return True

    except Exception:

        return False


def collection_has_documents(collection_name):

    try:

        collection = client.get_collection(
            name=collection_name
        )

        return collection.count() > 0

    except Exception:

        return False


def get_pdf_metadata(collection_name):

    try:

        collection = client.get_collection(
            name=collection_name
        )

        metadata = collection.metadata or {}

        pdf_data = metadata.get(
            "pdf_files_json",
            "[]"
        )

        return json.loads(pdf_data)

    except Exception as e:

        print(f"PDF metadata retrieval error: {e}")

        return []


def save_pdf_metadata(collection, pdf_details):

    metadata = collection.metadata or {}

    metadata["app"] = "pdf_rag"

    metadata["last_accessed"] = time.time()

    metadata["pdf_files_json"] = json.dumps(
        pdf_details
    )

    collection.modify(
        metadata=metadata
    )


def touch_collection(collection_name):

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "app": "pdf_rag",
            "last_accessed": time.time()
        }
    )

    metadata = collection.metadata or {}

    metadata["app"] = "pdf_rag"

    metadata["last_accessed"] = time.time()

    collection.modify(
        metadata=metadata
    )


# --------------------------------
# REMOVE INDIVIDUAL PDF
# --------------------------------

def remove_pdf(collection_name, pdf_hash):

    try:

        collection = client.get_collection(
            name=collection_name
        )

        pdf_details = get_pdf_metadata(
            collection_name
        )

        matching_pdf = next(
            (
                pdf for pdf in pdf_details
                if pdf.get("hash") == pdf_hash
            ),
            None
        )

        if matching_pdf is None:

            return {
                "success": False,
                "message": "PDF not found."
            }

        # Find chunks belonging to this PDF
        existing_data = collection.get(
            where={
                "pdf_hash": pdf_hash
            }
        )

        existing_ids = existing_data.get(
            "ids",
            []
        )

        if existing_ids:

            collection.delete(
                ids=existing_ids
            )

        # Remove PDF from metadata
        updated_details = [
            pdf for pdf in pdf_details
            if pdf.get("hash") != pdf_hash
        ]

        save_pdf_metadata(
            collection,
            updated_details
        )

        return {
            "success": True,
            "message": f"{matching_pdf['name']} removed successfully.",
            "pdf_details": updated_details
        }

    except Exception as e:

        print(f"PDF removal error: {e}")

        return {
            "success": False,
            "message": f"Could not remove PDF: {str(e)}"
        }


# --------------------------------
# AUTOMATIC COLLECTION CLEANUP
# --------------------------------

def cleanup_expired_collections(
        ttl_hours=24,
        exclude_collection=None
):

    current_time = time.time()

    expiry_seconds = ttl_hours * 60 * 60

    deleted_collections = []

    collections = client.list_collections()

    for collection in collections:

        name = (
            collection.name
            if hasattr(collection, "name")
            else collection
        )

        if not name.startswith("pdf_"):
            continue

        if name == exclude_collection:
            continue

        current_collection = client.get_collection(
            name=name
        )

        metadata = current_collection.metadata or {}

        last_accessed = metadata.get("last_accessed")

        if last_accessed is None:

            metadata["app"] = "pdf_rag"

            metadata["last_accessed"] = current_time

            current_collection.modify(
                metadata=metadata
            )

            continue

        if current_time - last_accessed > expiry_seconds:

            client.delete_collection(
                name=name
            )

            deleted_collections.append(name)

    print(
        "Deleted expired collections:",
        deleted_collections
    )

    return deleted_collections


# --------------------------------
# PDF PROCESSING
# --------------------------------

def process_pdfs(
        pdf_paths,
        collection_name,
        replace_existing=True
):

    successful_pdfs = []

    skipped_pdfs = []

    pdf_details = []

    all_chunks = []

    seen_hashes = set()

    # --------------------------------
    # GET EXISTING PDF INFORMATION
    # --------------------------------

    existing_details = get_pdf_metadata(
        collection_name
    )

    existing_hashes = {
        pdf.get("hash")
        for pdf in existing_details
        if pdf.get("hash")
    }

    # --------------------------------
    # VALIDATE EACH PDF
    # --------------------------------

    for pdf_path in pdf_paths:

        pdf_path = Path(pdf_path)

        try:

            if not pdf_path.exists() or not pdf_path.is_file():

                skipped_pdfs.append(
                    (pdf_path.name, "File not found")
                )

                continue

            if pdf_path.suffix.lower() != ".pdf":

                skipped_pdfs.append(
                    (pdf_path.name, "Not a PDF file")
                )

                continue

            if pdf_path.stat().st_size == 0:

                skipped_pdfs.append(
                    (pdf_path.name, "Empty PDF")
                )

                continue

            # Calculate SHA-256 hash
            file_hash = hashlib.sha256(
                pdf_path.read_bytes()
            ).hexdigest()

            # Duplicate within current upload
            if file_hash in seen_hashes:

                skipped_pdfs.append(
                    (pdf_path.name, "Duplicate PDF in upload")
                )

                continue

            seen_hashes.add(file_hash)

            # Duplicate against existing collection
            if not replace_existing and file_hash in existing_hashes:

                skipped_pdfs.append(
                    (pdf_path.name, "PDF already uploaded")
                )

                continue

            # --------------------------------
            # CHECK PDF STRUCTURE
            # --------------------------------

            try:

                with pymupdf.open(pdf_path) as document:

                    page_count = document.page_count

                    if page_count == 0:

                        skipped_pdfs.append(
                            (pdf_path.name, "Invalid PDF: no pages")
                        )

                        continue

            except Exception:

                skipped_pdfs.append(
                    (pdf_path.name, "Corrupted or unreadable PDF")
                )

                continue

            # --------------------------------
            # EXTRACT TEXT
            # --------------------------------

            try:

                pages = extract_text(str(pdf_path))

            except Exception:

                skipped_pdfs.append(
                    (pdf_path.name, "Text extraction failed")
                )

                continue

            if not pages:

                skipped_pdfs.append(
                    (pdf_path.name, "No extractable text")
                )

                continue

            extracted_text = ""

            for page in pages:

                if isinstance(page, dict):

                    extracted_text += page.get("text", "")

                else:

                    extracted_text += str(page)

            if not extracted_text.strip():

                skipped_pdfs.append(
                    (pdf_path.name, "No extractable text")
                )

                continue

            # --------------------------------
            # CREATE CHUNKS
            # --------------------------------

            try:

                chunks = create_chunks(pages)

            except Exception:

                skipped_pdfs.append(
                    (pdf_path.name, "Chunk creation failed")
                )

                continue

            if not chunks:

                skipped_pdfs.append(
                    (pdf_path.name, "No chunks created")
                )

                continue

            # Attach PDF information to chunks
            for chunk in chunks:

                if isinstance(chunk, dict):

                    chunk_data = chunk.copy()

                    chunk_data["source"] = pdf_path.name

                else:

                    chunk_data = {
                        "text": str(chunk),
                        "page": 0,
                        "source": pdf_path.name
                    }

                chunk_data["pdf_hash"] = file_hash

                all_chunks.append(chunk_data)

            successful_pdfs.append(pdf_path.name)

            pdf_details.append({

                "name": pdf_path.name,

                "hash": file_hash,

                "page_count": page_count,

                "chunk_count": len(chunks),

                "status": "Processed"

            })

        except Exception as e:

            skipped_pdfs.append(
                (pdf_path.name, f"Processing error: {str(e)}")
            )

    # --------------------------------
    # CHECK CHUNKS
    # --------------------------------

    if not all_chunks:

        return {
            "success": False,
            "message": "No new valid PDF content could be processed.",
            "successful_pdfs": successful_pdfs,
            "skipped_pdfs": skipped_pdfs,
            "total_chunks": 0,
            "pdf_details": []
        }

    # --------------------------------
    # PREPARE DOCUMENTS AND METADATA
    # --------------------------------

    documents = []

    metadatas = []

    chunk_ids = []

    for chunk in all_chunks:

        if isinstance(chunk, dict):

            text = chunk.get(
                "text",
                chunk.get("content", "")
            )

            page_number = chunk.get(
                "page",
                chunk.get("page_number", 0)
            )

            source = chunk.get(
                "source",
                "Unknown"
            )

            pdf_hash = chunk.get(
                "pdf_hash",
                ""
            )

        else:

            text = str(chunk)

            page_number = 0

            source = "Unknown"

            pdf_hash = ""

        if not str(text).strip():
            continue

        documents.append(str(text))

        metadatas.append({

            "page": int(page_number)
            if str(page_number).isdigit() else 0,

            "source": str(source),

            "pdf_hash": str(pdf_hash)

        })

        # Unique ID for every chunk
        chunk_ids.append(
            f"{pdf_hash}_{len(chunk_ids)}"
        )

    if not documents:

        return {
            "success": False,
            "message": "No usable text chunks were created.",
            "successful_pdfs": successful_pdfs,
            "skipped_pdfs": skipped_pdfs,
            "total_chunks": 0,
            "pdf_details": []
        }

    # --------------------------------
    # GENERATE EMBEDDINGS
    # --------------------------------

    try:

        model = get_embedding_model()

        embeddings = model.encode(
            documents,
            convert_to_numpy=True
        )

    except Exception as e:

        return {
            "success": False,
            "message": f"Embedding generation failed: {str(e)}",
            "successful_pdfs": successful_pdfs,
            "skipped_pdfs": skipped_pdfs,
            "total_chunks": 0,
            "pdf_details": []
        }

    # --------------------------------
    # STORE IN CHROMADB
    # --------------------------------

    try:

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "app": "pdf_rag",
                "last_accessed": time.time()
            }
        )

        # Replace mode: remove previous documents
        if replace_existing:

            existing_data = collection.get()

            existing_ids = existing_data.get(
                "ids",
                []
            )

            if existing_ids:

                collection.delete(
                    ids=existing_ids
                )

            updated_details = pdf_details

        else:

            # Add mode: preserve existing PDFs
            updated_details = (
                    existing_details + pdf_details
            )

        collection.upsert(
            ids=chunk_ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        # Save updated PDF metadata
        save_pdf_metadata(
            collection,
            updated_details
        )

    except Exception as e:

        return {
            "success": False,
            "message": f"ChromaDB storage failed: {str(e)}",
            "successful_pdfs": successful_pdfs,
            "skipped_pdfs": skipped_pdfs,
            "total_chunks": 0,
            "pdf_details": []
        }

    # --------------------------------
    # RETURN RESULT
    # --------------------------------

    return {

        "success": True,

        "message": "PDFs processed successfully.",

        "successful_pdfs": successful_pdfs,

        "skipped_pdfs": skipped_pdfs,

        "total_chunks": len(documents),

        "pdf_details": updated_details

    }
