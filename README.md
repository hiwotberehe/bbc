# AI-Powered Library Management System Prototype

An intelligent, machine-learning-driven digital library platform built for lightweight, cloud-based emulation inside Google Colab using a unified Python SQLite database backend paired with a dynamic Streamlit frontend.

## 🗂️ Core Architecture Footprint
* **Frontend UI Framework:** Streamlit (integrated custom HTML/CSS telemetry formatting blocks).
* **Backend Execution Model:** Native Python 3 multi-layered workflow logic routing.
* **Database Engine:** Local serverless SQLite deployment database system (`library.db`).
* **Machine Learning Compute Architecture:** Scikit-Learn pipelines using TF-IDF token vectors, Cosine Similarity arrays for rank modeling, and Levenshtein Fuzzy String computation indices.

---

## 🚀 Step-by-Step Deployment Guide within Google Colab

Follow these execution protocols inside your Google Colab terminal cells to bring the application engine online.

### Step 1: Install Dependencies
Download and install the necessary system components listed inside `requirements.txt`:
```bash
!pip install streamlit pandas numpy scikit-learn fuzzywuzzy python-Levenshtein bcrypt plotly
!npm install -g localtunnel