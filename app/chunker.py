
def create_chunks(pages, chunk_size=200, overlap=30):

    all_chunks = []

    for page in pages:

        words = page["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)

            all_chunks.append({
                "page": page["page"],
                "text": chunk_text
            })

            start += chunk_size - overlap

    return all_chunks






if __name__ == "__main__":

    project_folder = Path(__file__).resolve().parent.parent
    pdf_path = project_folder / "documents" / "sample.pdf"

    pages = extract_text(pdf_path)
    print("Pages extracted:", len(pages))
    print("Page data:", pages)

    chunks = create_chunks(pages)

    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1} | Page {chunk['page']}")
        print(chunk["text"])
