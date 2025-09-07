from langchain_community.document_loaders import PyPDFLoader
import fitz
import os
import base64

def image_to_base64(image_path):
    """
    Convert an image file to a base64 string.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def load_pdf(file_path, image_output_dir="extracted_images"):
    """
    Load a PDF file, extract text and images, and return a list of documents with metadata.
    """

    # Load text pages
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Extract images
    os.makedirs(image_output_dir, exist_ok=True)
    docs = fitz.open(file_path)
    new_docs = []

    for page_index, page in enumerate(pages):
        for img_index, img in enumerate(docs.get_page_images(page_index)):
            xref = img[0]
            base_image = docs.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_filename = f"page{page_index+1}_img{img_index+1}.{image_ext}"
            image_path = os.path.join(image_output_dir, image_filename)

            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            image_b64 = image_to_base64(image_path)

            new_docs.append(
                type(page)(  
                    page_content=page.page_content,
                    metadata={
                        **page.metadata,
                        "image_path": image_path,
                        "image_base64": image_b64
                    }
                )
            )

    docs.close()

    return new_docs