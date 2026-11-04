from sentence_transformers import SentenceTransformer
from pdf_loader import extract_text
from chunker import create_chunks
from pathlib import Path


# 1. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Extract PDF text
project_folder = Path(__file__).resolve().parent.parent
pdf_path = project_folder / "documents" / "sample.pdf"

pages = extract_text(pdf_path)


# 3. Create chunks
chunks = create_chunks(pages)


# 4. Get text from each chunk
chunk_texts = [chunk["text"] for chunk in chunks]


# 5. Generate embeddings
embeddings = model.encode(chunk_texts, convert_to_numpy=True)


# 6. Display results
print("Total chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

for i, embedding in enumerate(embeddings):

    print(f"\nChunk {i + 1} | Page {chunks[i]['page']}")
    print("Embedding:", embedding[:5])
    print("Vector dimensions:", len(embedding))