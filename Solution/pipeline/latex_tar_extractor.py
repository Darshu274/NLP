"""Hybrid evaluation pipeline: function made with the help of ChatGPT"""

import os, re, tarfile, subprocess, shutil
from pathlib import Path

QUANTIKZ_PATTERNS = [
    r"\\begin\{quantikz\}.*?\\end\{quantikz\}",
    r"\\Qcircuit.*?\}",  # qcircuit macro blocks (rough but useful)
]

def safe_extract_tar(tar_path: str, out_dir: str):
    """
    Safely extract a TAR archive into a target directory.

    The function creates the output directory if it does not exist
    and extracts all contents of the archive.

    Parameters
    ----------
    tar_path : str
        Path to the TAR archive.
    out_dir : str
        Directory where the archive contents will be extracted.
    """
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(out_dir)

# IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"}

# def extract_q_images_from_tar(tar_path: str, out_dir: str):
#     """
#     Extract only image files whose filenames start with 'q' or 'Q'
#     from a LaTeX source tar archive.
#     """
#     os.makedirs(out_dir, exist_ok=True)

#     with tarfile.open(tar_path, "r:*") as tar:
#         for member in tar.getmembers():
#             if not member.isfile():
#                 continue

#             name = os.path.basename(member.name)
#             if not name.lower().startswith("q"):
#                 continue

#             if Path(name).suffix.lower() not in IMAGE_EXTS:
#                 continue

#             src = tar.extractfile(member)
#             if src is None:
#                 continue

#             with open(os.path.join(out_dir, name), "wb") as f:
#                 shutil.copyfileobj(src, f)

# def collect_q_images_from_extracted_tree(root_dir: str, out_dir: str):
#     root_dir = Path(root_dir)
#     out_dir = Path(out_dir)
#     out_dir.mkdir(exist_ok=True)

#     for path in root_dir.rglob("*"):
#         if not path.is_file():
#             continue

#         # Filename must start with Q/q
#         if not path.name.lower().startswith("q"):
#             continue

#         # Must be image-like
#         if path.suffix.lower() not in IMAGE_EXTS:
#             continue

#         print("[Q-IMAGE FOUND]", path)

#         # Avoid overwriting
#         dest = out_dir / f"{path.parent.name}_{path.name}"
#         shutil.copy(path, dest)

def find_tex_files(root_dir: str):
    """
    Recursively find all LaTeX source files in a directory.

    Parameters
    ----------
    root_dir : str
        Root directory to search.

    Returns
    -------
    list of str
        List of paths to `.tex` files.
    """
    return [str(p) for p in Path(root_dir).rglob("*.tex")]

def extract_quantikz_blocks(tex_text: str):
    """
    Extract quantikz and qcircuit code blocks from LaTeX text.

    The function searches for known quantum circuit environments
    and macros, removes duplicates, and filters out very small blocks.

    Parameters
    ----------
    tex_text : str
        Full LaTeX document text.

    Returns
    -------
    list of str
        List of unique quantum circuit LaTeX blocks.
    """
    blocks = []
    for pat in QUANTIKZ_PATTERNS:
        blocks += re.findall(pat, tex_text, flags=re.DOTALL)
    # de-duplicate while preserving order
    seen, out = set(), []
    for b in blocks:
        key = re.sub(r"\s+", " ", b.strip())
        if key not in seen and len(key) > 30:
            seen.add(key)
            out.append(b)
    return out

def build_standalone_tex(body: str) -> str:
    """
    Wrap LaTeX code in a standalone document template.

    This allows individual quantikz or qcircuit blocks to be
    compiled independently into PDF figures.

    Parameters
    ----------
    body : str
        LaTeX code representing a quantum circuit.

    Returns
    -------
    str
        Complete standalone LaTeX document.
    """
    # includes quantikz if available; qcircuit often works with qcircuit package if present
    return r"""
\documentclass[border=2pt]{standalone}
\usepackage{braket}
\usepackage{amsmath}
\usepackage{tikz}
\usepackage{quantikz}
\begin{document}
""" + "\n" + body + "\n" + r"\end{document}"

def run_cmd(cmd, cwd):
    """
    Execute a shell command in a specified working directory.

    Parameters
    ----------
    cmd : list of str
        Command and arguments to execute.
    cwd : str
        Working directory for the command.

    Returns
    -------
    returncode : int
        Exit status of the command.
    stdout : str
        Standard output of the command.
    stderr : str
        Standard error output of the command.
    """
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.returncode, r.stdout, r.stderr

def compile_tex_to_pdf(tex_path: str, workdir: str):
    """
    Compile a LaTeX file into a PDF using pdflatex.

    The compilation is run twice to ensure stable references
    and layout.

    Parameters
    ----------
    tex_path : str
        Path to the LaTeX file.
    workdir : str
        Directory in which pdflatex will be executed.

    Raises
    ------
    RuntimeError
        If pdflatex fails during compilation.
    """
    # run pdflatex twice for stability
    for _ in range(2):
        code, out, err = run_cmd(["pdflatex", "-interaction=nonstopmode", os.path.basename(tex_path)], cwd=workdir)
        if code != 0:
            raise RuntimeError(f"pdflatex failed:\n{err[-1200:]}")

def pdf_to_png(pdf_path: str, png_path: str, workdir: str):
    """
    Convert a PDF file into a PNG image using pdftocairo.

    Parameters
    ----------
    pdf_path : str
        Path to the input PDF file.
    png_path : str
        Destination path for the output PNG image.
    workdir : str
        Working directory where conversion is executed.

    Raises
    ------
    RuntimeError
        If the PDF-to-PNG conversion fails.
    """
    # pdftocairo -singlefile -png input.pdf outprefix
    outprefix = os.path.splitext(os.path.basename(png_path))[0]
    code, out, err = run_cmd(["pdftocairo", "-singlefile", "-png", os.path.basename(pdf_path), outprefix], cwd=workdir)
    if code != 0:
        raise RuntimeError(f"pdftocairo failed:\n{err[-1200:]}")
    produced = os.path.join(workdir, outprefix + ".png")
    shutil.move(produced, png_path)

def render_blocks_to_png(arxiv_id: str, tar_extract_dir: str, workdir: str):
    """
    Render quantum circuit LaTeX blocks into PNG images.

    The function searches extracted LaTeX source files for quantikz or
    qcircuit environments, compiles each detected block into a standalone
    PDF, converts it to a PNG image, and returns structured metadata for
    each rendered figure.

    Parameters
    ----------
    arxiv_id : str
        Identifier of the paper used for naming output files.
    tar_extract_dir : str
        Directory containing extracted LaTeX source files.
    workdir : str
        Working directory for LaTeX compilation and image generation.

    Returns
    -------
    list of dict
        List of rendered figure metadata dictionaries with the keys:
        - ``image_path`` : str
            Path to the generated PNG image.
        - ``page`` : int
            Always ``-1`` since figures are generated from LaTeX source.
        - ``figure`` : int
            Index of the rendered figure.
        - ``caption_text`` : str
            LaTeX source code of the quantum circuit block.
        - ``evidence`` : str
            Evidence tag indicating LaTeX-based extraction
            (``'latex_quantikz'``).
    """
    tex_files = find_tex_files(tar_extract_dir)
    candidates = []
    fig_index = 0

    for tf in tex_files:
        try:
            txt = Path(tf).read_text(errors="ignore")
        except Exception:
            continue

        blocks = extract_quantikz_blocks(txt)
        for b in blocks:
            fig_index += 1
            local_dir = os.path.join(workdir, arxiv_id, f"latexfig_{fig_index:04d}")
            os.makedirs(local_dir, exist_ok=True)

            standalone = build_standalone_tex(b)
            tex_out = os.path.join(local_dir, "fig.tex")
            Path(tex_out).write_text(standalone)

            # compile + convert
            compile_tex_to_pdf(tex_out, local_dir)
            pdf_out = os.path.join(local_dir, "fig.pdf")

            png_name = f"{arxiv_id}_p-1_fig{fig_index}_latex.png"
            png_out = os.path.join(workdir, png_name)
            pdf_to_png(pdf_out, png_out, local_dir)

            # caption_text: you can enrich later by searching surrounding lines in tf;
            # for now use block text (still useful for gates extraction)
            candidates.append({
                "image_path": png_out,
                "page": -1,
                "figure": fig_index,
                "caption_text": b,
                "evidence": "latex_quantikz"
            })
    return candidates
