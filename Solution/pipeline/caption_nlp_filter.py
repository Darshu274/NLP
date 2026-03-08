"""Hybrid evaluation pipeline: function made with the help of ChatGPT"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

REFERENCE_TEXTS = [
    "quantum circuit diagram",
    "quantum gate based circuit",
    "quantikz circuit",
    "qcircuit diagram"
]

REF_VEC = model.encode(REFERENCE_TEXTS).mean(axis=0).reshape(1, -1)

# # KEYWORD TIERS
# # Strong LaTeX / circuit evidence
# HARD_KEYWORDS = [
#     "quantikz", "qcircuit",
#     "\\gate", "\\ctrl", "\\targ", "\\lstick", "\\rstick",
#     "cnot", "cx", "cz", "ccx", "toffoli", "swap",
#     "\\meter", "\\measure", "\\qw", "\\qwx"
# ]

# # Common quantum gate names
# GATE_KEYWORDS = [
#     "hadamard", "pauli",
#     "rx", "ry", "rz", "u3", "u2", "u1",
#     "x gate", "y gate", "z gate",
#     "h gate", "t gate", "s gate",
#     "controlled gate",
#     "rotation gate",
#     "phase gate",
#     "measure", "measurement"
# ]

# # Structural words (weak alone)
# STRUCTURAL_KEYWORDS = [
#     "qubit", "qubits",
#     "register", "quantum register",
#     "wire", "wires",
#     "controlled",
#     "entanglement"
# ]

# # ===============================
# # KEYWORD LOGIC
# # ===============================
# def keyword_hit(text: str) -> bool:
#     """
#     Hierarchical keyword check to avoid false positives.
#     """
#     t = text.lower()

#     # Tier 1: hard evidence
#     if any(k in t for k in HARD_KEYWORDS):
#         return True

#     # Tier 2: gate names
#     if any(k in t for k in GATE_KEYWORDS):
#         return True

#     # Tier 3: structure + circuit context
#     if ("circuit" in t or "diagram" in t) and any(k in t for k in STRUCTURAL_KEYWORDS):
#         return True

#     return False

# ===============================
# KEYWORD LOGIC
# ===============================
# def keyword_hit(text: str) -> bool:
#     """
#     Hierarchical keyword check to avoid false positives.
#     """
#     t = text.lower()

#     # Tier 1: hard evidence
#     if any(k in t for k in HARD_KEYWORDS):
#         return True

#     # Tier 2: gate names
#     if any(k in t for k in GATE_KEYWORDS):
#         return True

#     # Tier 3: structure + circuit context
#     if ("circuit" in t or "diagram" in t) and any(k in t for k in STRUCTURAL_KEYWORDS):
#         return True

#     return False

def is_quantum_circuit(caption, threshold=0.38):
    """
    Determine whether a given image caption describes a quantum circuit.

    The function computes a semantic similarity score between the input
    caption and a reference embedding representing typical quantum circuit
    descriptions. In addition, it performs a keyword-based check to reduce
    false positives.

    Parameters
    ----------
    caption : str
        The textual description or caption associated with an image.
    threshold : float, optional
        Minimum cosine similarity score required to classify the caption
        as a quantum circuit. Default is 0.38.

    Returns
    -------
    is_circuit : bool
        True if the caption is classified as a quantum circuit, otherwise False.
    score : float
        Cosine similarity score between the caption embedding and the
        reference quantum circuit embedding.
    """
    vec = model.encode([caption])
    score = cosine_similarity(vec, REF_VEC)[0][0]

    keywords = ["circuit", "gate", "qubit", "cnot", "quantikz"]
    keyword_hit = any(k in caption.lower() for k in keywords)

    return score >= threshold and keyword_hit, score
