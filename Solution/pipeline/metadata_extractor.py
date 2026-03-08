# """Hybrid evaluation pipeline: function made with the help of ChatGPT"""
# import re

# def extract_gates(text):
#     """
#     Extract quantum gate names mentioned in a text string.

#     The function searches the input text for occurrences of known
#     quantum gate identifiers using regular expressions and returns
#     a unique list of detected gates.

#     Parameters
#     ----------
#     text : str
#         Input text from which quantum gate names will be extracted.

#     Returns
#     -------
#     list of str
#         List of unique quantum gate names found in the text.
#     """
#     gates = ["H", "X", "Y", "Z", "CNOT", "CX", "CZ", "SWAP", "TOFFOLI"]
#     return list({g for g in gates if re.search(rf"\b{g}\b", text.upper())})

# def extract_algorithm(text):
#     """
#     Identify a quantum algorithm referenced in a text string.

#     The function checks for known keywords associated with common
#     quantum algorithms and returns the corresponding algorithm name.

#     Parameters
#     ----------
#     text : str
#         Input text from which the algorithm name will be identified.

#     Returns
#     -------
#     str
#         Name of the detected quantum algorithm, or ``"unknown"`` if
#         no known algorithm is found.
#     """
#     algos = {
#         "shor": "Shor algorithm",
#         "grover": "Grover algorithm",
#         "qft": "Quantum Fourier Transform"
#     }
#     for k, v in algos.items():
#         if k in text.lower():
#             return v
#     return "unknown"

import re

def extract_gates(text):
    """
    Extract quantum gate names mentioned in a text string.

    The function searches the input text for occurrences of known
    quantum gate identifiers using regular expressions and returns
    a unique list of detected gates.

    Parameters
    ----------
    text : str
        Input text from which quantum gate names will be extracted.

    Returns
    -------
    list of str
        List of unique quantum gate names found in the text.
    """
    gates = [
        # Identity
        "I", "ID", "IDENTITY",

        # Pauli Gates
        "X", "Y", "Z",

        # Clifford Gates
        "H",                 # Hadamard
        "S", "SDG",           # Phase, Phase-dagger
        "CX", "CNOT",         # Controlled-NOT
        "CY", "CZ", "CH",     # Controlled Pauli / Hadamard

        # Non-Clifford Gates
        "T", "TDG",           # T gate and adjoint

        # Rotation Gates
        "RX", "RY", "RZ",
        "U", "U1", "U2", "U3",

        # Square-root / Fractional Gates
        "SX", "SXDG",
        "SQRTX", "SQRTY", "SQRTZ",

        # Controlled Rotation Gates
        "CRX", "CRY", "CRZ",
        "CU", "CU1", "CU2", "CU3",

        # Multi-Controlled Gates
        "CCX", "TOFFOLI",
        "CCZ",
        "MCX", "MCZ", "MCU",

        # Swap Family
        "SWAP",
        "CSWAP", "FREDKIN",
        "ISWAP",
        "SQRTSWAP",

        # Entangling / Interaction Gates
        "XX", "YY", "ZZ",
        "RXX", "RYY", "RZZ",
        "MS",                # Mølmer–Sørensen
        "FSIM",

        # Measurement Operations
        "MEASURE",
        "MEASURE_X",
        "MEASURE_Y",
        "MEASURE_Z",

        # State Preparation / Reset
        "RESET",
        "PREPARE",
        "STATE_PREP",

        # Algorithmic / Abstract Gates
        "ORACLE",
        "DIFFUSER",
        "QFT",
        "IQFT",
        "GROVER",

        # Noise / Channel Models
        "DEPOLARIZING",
        "AMPLITUDE_DAMPING",
        "PHASE_DAMPING",
        "KRAUS",

        # Hardware-Specific Gates
        "ECR",               # IBM Echoed Cross-Resonance
        "RZZ",
        "XY",
    ]

    return list({g for g in gates if re.search(rf"\b{g}\b", text.upper())})

def extract_algorithm(text: str) -> str:
    if not text:
        return "Unknown / not specified"

    t = text.lower()

    # Level 1: explicit named algorithms
    NAMED_ALGOS = {
    # Factoring / search
    "shor": "Shor algorithm",
    "grover": "Grover algorithm",

    # Fourier / phase
    "quantum fourier": "Quantum Fourier Transform",
    "qft": "Quantum Fourier Transform",
    "phase estimation": "Quantum Phase Estimation",
    "qpe": "Quantum Phase Estimation",

    # Near-term / variational (explicitly named in many papers)
    "vqe": "Variational Quantum Eigensolver",
    "variational quantum eigensolver": "Variational Quantum Eigensolver",
    "qaoa": "Quantum Approximate Optimization Algorithm",
    "quantum approximate optimization": "Quantum Approximate Optimization Algorithm",

    # Simulation
    "quantum simulation": "Quantum simulation",
    "hamiltonian simulation": "Hamiltonian simulation",

    # Error correction (only when explicitly stated)
    "surface code": "Surface code",
    "stabilizer code": "Stabilizer code",

    # Benchmarking (explicit)
    "randomized benchmarking": "Randomized benchmarking"
    }


    for k, v in NAMED_ALGOS.items():
        if k in t:
            return v

    # Level 2: task classes
    TASK_CLASSES = {
        "Error correction": [
            "error correction", "surface code", "stabilizer", "syndrome"
        ],
        "State preparation": [
            "state preparation", "bell state", "ghz", "entangled state"
        ],
        "Variational algorithm": [
            "variational", "ansatz", "vqe", "qaoa", "parameterized"
        ],
        "Measurement circuit": [
            "measurement", "readout", "expectation value"
        ],
        "Benchmark / example circuit": [
            "example circuit", "toy model", "illustrative"
        ],
        "Compilation / optimization": [
            "transpile", "mapping", "compiled", "optimization"
        ]
    }

    for label, keywords in TASK_CLASSES.items():
        if any(k in t for k in keywords):
            return label

    # Level 3: unknown
    return "Unknown / not specified"