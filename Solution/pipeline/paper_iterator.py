
"""Hybrid evaluation pipeline: function made with the help of ChatGPT"""
def iterate_papers(paper_list_file):
    """
    Iterate over a list of paper identifiers stored in a text file.

    The function reads a file line by line and yields non-empty
    paper identifiers (e.g., arXiv IDs).

    Parameters
    ----------
    paper_list_file : str
        Path to the text file containing paper identifiers, one per line.

    Yields
    ------
    str
        A single paper identifier from the file.
    """
    with open(paper_list_file) as f:
        for line in f:
            arxiv_id = line.strip()
            if arxiv_id:
                yield arxiv_id

