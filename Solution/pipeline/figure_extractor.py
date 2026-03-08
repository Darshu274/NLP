"""Hybrid evaluation pipeline: function made with the help of ChatGPT"""

import fitz
import os

def extract_figures_from_pdf(arxiv_id, pdf_path, workdir):
    """
    Extract raster images from a PDF document and store them as image files.

    This function iterates through all pages of a given PDF file, extracts
    embedded images, converts them to RGB format when necessary, and saves
    them to disk. For each extracted image, associated metadata such as page
    number and surrounding page text is recorded.

    Parameters
    ----------
    arxiv_id : str
        Identifier of the paper (e.g., arXiv ID) used to generate unique
        image filenames.
    pdf_path : str
        Path to the input PDF document.
    workdir : str
        Directory where extracted images will be saved.

    Returns
    -------
    figures : list of dict
        A list of dictionaries, each containing metadata for an extracted
        figure. Each dictionary includes the following keys:
        - ``image_path`` : str
            File path of the saved image.
        - ``page`` : int
            Page number in the PDF where the image was found.
        - ``caption_text`` : str
            Text content extracted from the corresponding page.
    """
    doc = fitz.open(pdf_path)
    figures = []

    for page_no in range(len(doc)):
        page = doc[page_no]
        images = page.get_images(full=True)
        text = page.get_text()

        for idx, img in enumerate(images):
            xref = img[0]

            pix = fitz.Pixmap(doc, xref)

            # CRITICAL FIX
            if pix.colorspace is None:
                pix = None
                continue

            if pix.colorspace.n != 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            if pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            fname = f"{arxiv_id}_p{page_no+1}_{idx}.png"
            fpath = os.path.join(workdir, fname)

            pix.save(fpath)
            pix = None

            figures.append({
                "image_path": fpath,
                "page": page_no + 1,
                "caption_text": text
            })

    return figures
