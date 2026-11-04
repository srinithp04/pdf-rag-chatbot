
import pymupdf


def extract_text(pdf_path):

    # Open the PDF document
    document = pymupdf.open(pdf_path)

    extracted_pages = []

    # Read each page
    for page_number, page in enumerate(document):

        text = page.get_text()

        extracted_pages.append({
            "page": page_number + 1,
            "text": text
        })

    document.close()

    return extracted_pages


if __name__ == "__main__":

    pdf_path = r"C:\Users\ELCOT\Downloads\rag\pdf-rag-chatbot\documents\sample.pdf"

    pages = extract_text(pdf_path)


    print("PDF extraction completed!")
    print("--------------------------------")

    for page in pages:
        print(f"\nPage: {page['page']}")
        print(page["text"])
