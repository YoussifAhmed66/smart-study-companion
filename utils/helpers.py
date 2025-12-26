import fitz

def is_image(file_path):
    """Check if the file is an image based on its extension."""
    valid_extensions = ('.jpg', '.jpeg', '.png')
    return file_path.lower().endswith(valid_extensions)

def is_pdf(file_path):
    """Check if the file is a PDF based on its extension."""
    return file_path.lower().endswith('.pdf')

def convert_pdf_to_images(pdf_path):
    """
    Convert a PDF file into a list of image bytes (JPEG format).

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        list[bytes]: A list of byte objects, each representing a page as an image.
    """
    doc = fitz.open(pdf_path)
    images_bytes = []
    
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        images_bytes.append(pix.tobytes("jpg"))
        
    doc.close()
    return images_bytes