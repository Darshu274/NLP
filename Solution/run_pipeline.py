"""
Usage:
    python main.py

References:
        - https://arxiv.org/help/bulk_data_s3
        - https://arxiv.org/help/download
        - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        - https://github.com/pgf-tikz/pgf/tree/master/tex/latex/pgf
        - https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
        - https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
        - https://pymupdf.readthedocs.io/en/latest/recipes-ocr.html#how-to-ocr-an-image
        - Hybrid evaluation pipeline: function made with the help of ChatGPT
"""

import os
import json
import random
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pipeline.paper_iterator import iterate_papers
from pipeline.source_fetcher import fetch_arxiv_source, fetch_pdf
from pipeline.figure_extractor import extract_figures_from_pdf
from pipeline.caption_nlp_filter import is_quantum_circuit
from pipeline.cv_circuit_filter import looks_like_quantum_circuit
from pipeline.metadata_extractor import extract_gates, extract_algorithm
from pipeline.dataset_writer import write_json
from pipeline.latex_tar_extractor import safe_extract_tar, render_blocks_to_png
from pipeline.nlp_threshold_tuner import choose_threshold

# ===============================
# CONFIG
# ===============================
PAPER_LIST = "paper_list_38.txt"
IMAGES_DIR = "images_38"
WORKDIR = "workdir"
CSV_OUT = "paper_list_counts_38.csv"
JSON_OUT = "metadata_38.json"

TARGET = 250

# Threshold tuning cache
CACHE_DIR = "cache"
THRESH_FILE = os.path.join(CACHE_DIR, "threshold_38.txt")

# NLP tuning behavior
MIN_POS = 20      # from LaTeX (silver positives)
MIN_NEG = 60      # from PDF random negatives
TARGET_PREC = 0.90

# PDF candidate sampling for tuning
NEG_SAMPLES_PER_PAPER = 8

# ===============================
# HELPERS
# ===============================
def ensure_dirs():
    """
    Create required output and working directories if they do not exist.
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(WORKDIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def load_metadata():
    """
    Load existing metadata from disk if available.

    Returns
    -------
    dict
        Previously saved metadata dictionary, or an empty dictionary
        if no metadata file exists.
    """
    if os.path.exists(JSON_OUT):
        with open(JSON_OUT, "r") as f:
            return json.load(f)
    return {}

def atomic_write_csv(count_rows):
    """
    Write paper image counts to CSV in a single operation.

    Parameters
    ----------
    count_rows : list of dict
        List of rows containing paper identifiers and image counts.
    """
    df = pd.DataFrame(count_rows)
    df.to_csv(CSV_OUT, index=False)

def load_existing_counts():
    """
    Load previously stored per-paper image counts.

    Returns
    -------
    dict
        Mapping from arXiv ID to image count or empty string if not processed.
    """
    if not os.path.exists(CSV_OUT):
        return {}
    df = pd.read_csv(CSV_OUT, dtype={"arxiv_id": str})
    out = {}
    for _, r in df.iterrows():
        out[str(r["arxiv_id"]).strip()] = r.get("count", "")
    return out

def save_threshold(t):
    """
    Persist the tuned NLP threshold to disk.

    Parameters
    ----------
    t : float
        Threshold value to store.
    """
    with open(THRESH_FILE, "w") as f:
        f.write(str(float(t)))

def load_threshold():
    """
    Load a previously tuned NLP threshold from disk.

    Returns
    -------
    float or None
        Loaded threshold value, or None if not available.
    """
    if os.path.exists(THRESH_FILE):
        try:
            return float(Path(THRESH_FILE).read_text().strip())
        except Exception:
            return None
    return None

def get_score_only(text):
    """
    Compute the NLP similarity score for a caption without applying a threshold.

    Parameters
    ----------
    text : str
        Caption or textual description.

    Returns
    -------
    float
        Similarity score produced by the NLP model.
    """
    ok, score = is_quantum_circuit(text, threshold=-999.0)
    return score

def contains_gate_tokens(text):
    """
    Check whether a text contains explicit quantum gate indicators.

    Parameters
    ----------
    text : str
        Caption or LaTeX text.

    Returns
    -------
    bool
        True if gate-related tokens are present, otherwise False.
    """
    tokens = [
        "cnot", "cx", "cz", "swap", "toffoli", "ccx",
        "hadamard", "phase", "t gate", "s gate",
        "rx", "ry", "rz", "u3", "measure", "measurement",
        "\\gate", "\\ctrl", "\\targ", "\\lstick", "\\rstick",
        "quantikz", "qcircuit"
    ]
    tl = text.lower()
    return any(t in tl for t in tokens)

def move_or_copy(src, dst):
    """
    Move a file to a destination path, creating directories if required.

    Parameters
    ----------
    src : str
        Source file path.
    dst : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)

def clean_paper_workdir(arxiv_id):
    """
    Remove intermediate working directories for a paper.

    Parameters
    ----------
    arxiv_id : str
        Identifier of the arXiv paper.
    """
    # Optional: keeps your disk small. Comment this out if you want to keep artifacts.
    # Removes extracted latex folders for that paper.
    paper_dir = os.path.join(WORKDIR, arxiv_id)
    if os.path.isdir(paper_dir):
        shutil.rmtree(paper_dir, ignore_errors=True)

def build_counts_rows_in_order(paper_ids, existing_counts_map, updated_counts_map):
    """
    Build CSV rows in the original paper list order.

    Parameters
    ----------
    paper_ids : list of str
        Ordered list of paper identifiers.
    existing_counts_map : dict
        Previously stored counts.
    updated_counts_map : dict
        Counts updated during the current run.

    Returns
    -------
    list of dict
        Rows suitable for CSV export.
    """
    rows = []
    for pid in paper_ids:
        if pid in updated_counts_map:
            rows.append({"arxiv_id": pid, "count": updated_counts_map[pid]})
        else:
            # preserve old if exists, otherwise blank
            val = existing_counts_map.get(pid, "")
            rows.append({"arxiv_id": pid, "count": val})
    return rows

        # def clean_caption_text(text: str) -> str:
#     """
#     Extract the sentence that contains a figure reference
#     ('Figure' or 'Fig.', case-insensitive).
#     """
#     if not text:
#         return ""

#     # Normalize whitespace
#     text = text.replace("\n", " ").strip()

#     # Split into sentence-like chunks
#     sentences = text.split(".")

#     for s in sentences:
#         sl = s.lower()
#         if "figure" in sl or "fig." in sl or "fig " in sl:
#             return s.strip()

#     # Fallback: trim aggressively
#     return text[:300]


# def looks_like_training_plot_caption(text: str) -> bool:
#     """
#     Return True if caption likely describes a training/performance plot.
#     """
#     if not text:
#         return False

#     NEGATIVE_WORDS = [
#         "loss", "error", "accuracy", "epoch", "training",
#         "validation", "test accuracy", "runtime", "time",
#         "convergence", "benchmark", "performance", "speedup", "operators",
#         "variance", "value", "iteration", "probabilities", "Hz", "Estimate",
#         "metrics", "steps", "shots", "edges", "violation", "fidelity",
#         "distribution", "histogram", "scatter", "plot", "optimizer",
#         "iterations"
#     ]

#     t = text.lower()
#     return any(w in t for w in NEGATIVE_WORDS)
# ===============================
# THRESHOLD TUNING
# ===============================
class ThresholdTuner:
    """
    Adaptive threshold tuner for NLP-based circuit classification.

    The tuner collects positive and negative scores and determines
    an optimal decision threshold that satisfies a target precision
    constraint.
    """
    def __init__(self):
        self.pos_scores = []
        self.neg_scores = []
        self.threshold = load_threshold()

    def ready(self):
        """
        Check whether a tuned threshold is already available.

        Returns
        -------
        bool
            True if a cached threshold exists, otherwise False.
        """
        return self.threshold is not None

    def add_positive(self, score):
        """
        Add a positive example score.

        Parameters
        ----------
        score : float
            NLP similarity score.
        """
        self.pos_scores.append(float(score))

    def add_negative(self, score):
        """
        Add a negative example score.

        Parameters
        ----------
        score : float
            NLP similarity score.
        """
        self.neg_scores.append(float(score))

    def maybe_tune(self):
        """
        Tune the threshold if sufficient data is available.

        Returns
        -------
        float or None
            Tuned threshold value if available.
        """
        if self.threshold is not None:
            return self.threshold

        if len(self.pos_scores) >= MIN_POS and len(self.neg_scores) >= MIN_NEG:
            scores = self.pos_scores + self.neg_scores
            labels = [1] * len(self.pos_scores) + [0] * len(self.neg_scores)

            t, p, r, f1 = choose_threshold(scores, labels, target_precision=TARGET_PREC)
            self.threshold = float(t)
            save_threshold(self.threshold)

            print("\n Tuned NLP threshold")
            print(f" threshold={self.threshold:.4f}  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}\n")

        return self.threshold

        # def has_figure_reference(text: str) -> bool:
        #     """
        #     Check whether caption text contains a figure reference
        #     (case-insensitive).
        #     """
        #     if not text:
        #         return False
        #     t = text.lower()
        #     return ("figure" in t) or ("fig." in t) or ("fig " in t)
# ===============================
# MAIN PIPELINE
# ===============================
def main():
    """
    Execute the full dataset construction pipeline.

    The pipeline processes a list of arXiv papers, extracts quantum
    circuit figures from LaTeX sources or PDFs, applies NLP and
    computer vision filtering, tunes decision thresholds, and
    incrementally builds a curated dataset with metadata.
    """
    ensure_dirs()

    # Read ordered paper list
    with open(PAPER_LIST, "r") as f:
        paper_ids = [line.strip() for line in f if line.strip()]

    # Resume support
    existing_counts = load_existing_counts()
    metadata = load_metadata()
    total = len(metadata)  # already accepted images across runs

    print(f" Resume: metadata has {total} accepted images already.")

    tuner = ThresholdTuner()
    if tuner.ready():
        print(f"Using cached NLP threshold: {tuner.threshold:.4f}")
    else:
        print("ℹ  No cached threshold yet. Will tune automatically using early papers (LaTeX positives + PDF negatives).")

    updated_counts = {}

    # Process papers in order
    for arxiv_id in tqdm(paper_ids, desc="Processing papers"):
        # If we reached target, leave remaining blank (unless already processed)
        if total >= TARGET:
            # If not processed earlier, keep blank
            if arxiv_id not in existing_counts and arxiv_id not in updated_counts:
                updated_counts[arxiv_id] = ""
            continue

        # If already processed in previous run (count is numeric), skip
        prev = existing_counts.get(arxiv_id, "")
        if prev != "" and str(prev).strip().isdigit():
            # Do not change counts; do not reprocess
            continue

        accepted_this_paper = 0

        # ---------- (1) Try LaTeX TAR first ----------
        tar_path = fetch_arxiv_source(arxiv_id, WORKDIR)
        latex_candidates = []
        if tar_path:
            extract_dir = os.path.join(WORKDIR, arxiv_id, "src")
            os.makedirs(extract_dir, exist_ok=True)
            try:
                safe_extract_tar(tar_path, extract_dir)
                latex_candidates = render_blocks_to_png(arxiv_id, extract_dir, WORKDIR)
            except Exception as e:
                print(f"\n LaTeX TAR extraction failed for {arxiv_id}: {e}")
                latex_candidates = []

        # Accept LaTeX candidates (high precision)
        for cand in latex_candidates:
            if total >= TARGET:
                break

            # Optionally run CV filter (usually true). If it fails, still accept because LaTeX evidence is strong.
            cv_ok = looks_like_quantum_circuit(cand["image_path"])
            # Score (for tuning + metadata)
            score = get_score_only(cand["caption_text"])
            tuner.add_positive(score)

            out_name = f"{arxiv_id}_p{cand['page']}_fig{cand['figure']}_latex.png"
            out_path = os.path.join(IMAGES_DIR, out_name)
            move_or_copy(cand["image_path"], out_path)

            caption_text = cand["caption_text"]

            metadata[out_name] = {
                "arxiv_id": arxiv_id,
                "page": int(cand["page"]),
                "figure": int(cand["figure"]),
                "quantum_gates": extract_gates(caption_text),
                "quantum_problem": extract_algorithm(caption_text),
                "descriptions": [caption_text],
                "text_positions": [(0, len(caption_text))],
                "evidence": "latex_quantikz",
                "nlp_score": float(score),
                "cv_ok": bool(cv_ok),
            }

            accepted_this_paper += 1
            total += 1

        # Tune threshold when possible (for PDF stage)
        thresh = tuner.maybe_tune()
        if thresh is None:
            # fallback if not tuned yet
            thresh = 0.45  # conservative default until tuning kicks in

        # ---------- (2) If LaTeX yielded nothing, fallback to PDF ----------
        if accepted_this_paper == 0:
            pdf_path = fetch_pdf(arxiv_id, WORKDIR)
            if not pdf_path:
                updated_counts[arxiv_id] = 0
                # write partial CSV+JSON
                rows = build_counts_rows_in_order(paper_ids, existing_counts, updated_counts)
                atomic_write_csv(rows)
                write_json(JSON_OUT, metadata)
                continue

            pdf_candidates = extract_figures_from_pdf(arxiv_id, pdf_path, WORKDIR)

            # Add some negatives for tuning (random subset)
            random.shuffle(pdf_candidates)
            for cand in pdf_candidates[:NEG_SAMPLES_PER_PAPER]:
                score = get_score_only(cand["caption_text"])
                tuner.add_negative(score)
            tuner.maybe_tune()
            thresh = tuner.threshold if tuner.threshold is not None else thresh

            # Now do real filtering: NLP + gate evidence + CV structure
            for i, cand in enumerate(pdf_candidates):
                if total >= TARGET:
                    break

                caption_text = cand["caption_text"]

                # Stronger NLP gate: needs score AND gate evidence
                ok, score = is_quantum_circuit(caption_text, threshold=thresh)
                if not ok:
                    # delete rejected file to save disk
                    try:
                        os.remove(cand["image_path"])
                    except Exception:
                        pass
                    continue

                if not contains_gate_tokens(caption_text):
                    try:
                        os.remove(cand["image_path"])
                    except Exception:
                        pass
                    continue

                if not looks_like_quantum_circuit(cand["image_path"]):
                    try:
                        os.remove(cand["image_path"])
                    except Exception:
                        pass
                    continue

                out_name = f"{arxiv_id}_p{cand['page']}_fig{i}_pdf.png"
                out_path = os.path.join(IMAGES_DIR, out_name)
                move_or_copy(cand["image_path"], out_path)

                metadata[out_name] = {
                    "arxiv_id": arxiv_id,
                    "page": int(cand["page"]),
                    "figure": int(i),
                    "quantum_gates": extract_gates(caption_text),
                    "quantum_problem": extract_algorithm(caption_text),
                    "descriptions": [caption_text],
                    "text_positions": [(0, len(caption_text))],
                    "evidence": "pdf_nlp_cv",
                    "nlp_score": float(score),
                    "cv_ok": True,
                    "threshold": float(thresh),
                }

                accepted_this_paper += 1
                total += 1

        updated_counts[arxiv_id] = accepted_this_paper

        # Write partial progress after each paper (safe resume)
        rows = build_counts_rows_in_order(paper_ids, existing_counts, updated_counts)
        atomic_write_csv(rows)
        write_json(JSON_OUT, metadata)

        # Optional cleanup (uncomment to reduce disk usage)
        # clean_paper_workdir(arxiv_id)

        print(f"\n {arxiv_id}: accepted {accepted_this_paper} images | total {total}/{TARGET} | threshold {thresh:.4f}")

    # Final write
    rows = build_counts_rows_in_order(paper_ids, existing_counts, updated_counts)
    atomic_write_csv(rows)
    write_json(JSON_OUT, metadata)

    print("\n DONE")
    print(f"Total accepted images: {len(metadata)}")
    print(f"Images folder: {IMAGES_DIR}")
    print(f"Counts CSV: {CSV_OUT}")
    print(f"Metadata JSON: {JSON_OUT}")
    if os.path.exists(THRESH_FILE):
        print(f"NLP threshold cached at: {THRESH_FILE}")

if __name__ == "__main__":
    main()


