"""Hybrid evaluation pipeline: function made with the help of ChatGPT"""

import requests
import os

def fetch_arxiv_source(arxiv_id, workdir):
    """
    Download the LaTeX source archive of an arXiv paper.

    The function retrieves the source files of a paper from arXiv in
    TAR format and stores them locally. If a valid archive already
    exists on disk, it is reused.

    Parameters
    ----------
    arxiv_id : str
        Identifier of the arXiv paper.
    workdir : str
        Directory where the source archive will be stored.

    Returns
    -------
    str or None
        Path to the downloaded TAR archive if successful, otherwise None.
    """
    url = f"https://export.arxiv.org/e-print/{arxiv_id}"
    out = os.path.join(workdir, f"{arxiv_id}.tar.gz")
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return out
    r = requests.get(url, timeout=30)
    if r.status_code == 200 and len(r.content) > 1000:
        with open(out, "wb") as f:
            f.write(r.content)
        return out
    return None

def fetch_pdf(arxiv_id, workdir):
    """
    Download the PDF version of an arXiv paper.

    Parameters
    ----------
    arxiv_id : str
        Identifier of the arXiv paper.
    workdir : str
        Directory where the PDF file will be saved.

    Returns
    -------
    str or None
        Path to the downloaded PDF file if successful, otherwise None.
    """
    pdf_url = f"https://export.arxiv.org/pdf/{arxiv_id}.pdf"
    pdf_path = os.path.join(workdir, f"{arxiv_id}.pdf")

    r = requests.get(pdf_url, timeout=20)
    if r.status_code == 200:
        with open(pdf_path, "wb") as f:
            f.write(r.content)
        return pdf_path

    return None
