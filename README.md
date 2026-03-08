# NLP Project – Quantum Circuit Dataset Construction Pipeline

This project implements an end-to-end pipeline to automatically extract, filter, and curate **quantum circuit images and metadata** from arXiv papers using **LaTeX parsing, NLP, and computer vision**.

---

## Setup Instructions
### Create a virtual environment
```bash
python -m venv .venv
```

---

### Activate the virtual environment

**On Windows**
```bash
.\.venv\Scripts\activate
```

**On macOS / Linux**
```bash
source .venv/bin/activate
```

---

### Install required packages
```bash
pip install -r requirements.txt
```

> ⚠️ Ensure **pdflatex** (TeX Live) and **pdftocairo** (Poppler) are installed and available in PATH.

---

## Project Structure

```text
NLP_PRO_LATEX_IMPROVEMENT/
│
├── pipeline/
│   ├── paper_iterator.py
│   ├── source_fetcher.py
│   ├── figure_extractor.py
│   ├── caption_nlp_filter.py
│   ├── cv_circuit_filter.py
│   ├── latex_tar_extractor.py
│   ├── metadata_extractor.py
│   ├── dataset_writer.py
│   ├── nlp_threshold_tuner.py
│
├── images_38/
├── workdir/
├── cache/
│
├── paper_list_38.txt
├── paper_list_counts_38.csv
├── metadata_38.json
├── requirements.txt
├── run_pipeline.py
```

---

## Running the Pipeline

Activate the virtual environment and run:

```bash
python run_pipeline.py
```

---

## Outputs

- **images_38/** – Curated quantum circuit images  
- **metadata_38.json** – Metadata (gates, algorithm, scores, evidence)  
- **paper_list_counts_38.csv** – Images per paper  

---

## Notes

- Supports resume across runs  
- Automatic NLP threshold tuning  
- High-precision LaTeX extraction + NLP/CV PDF fallback  

---

## Requirements

- Python 3.9+
- TeX Live (`pdflatex`)
- Poppler (`pdftocairo`)