# Wikipedia Search Engine - Information Retrieval Project

This project implements a search engine for the entire English Wikipedia corpus as part of the Information Retrieval course (Winter 2025-2026).
The engine is deployed on Google Cloud Platform (GCP), utilizing a distributed architecture with storage in a GCS Bucket and a compute instance running a Flask-based frontend.

## 🚀 Project Overview

The search engine is designed to retrieve relevant Wikipedia articles based on user queries. It supports free-text search and ranks results using a combination of **BM25** and **PageRank**.

### Key Features
* **Efficient Retrieval:** Handles queries over millions of documents with an average latency of < 2.5 seconds.
* **Ranking Algorithm:** Uses a weighted combination of content relevance (BM25) and static popularity (PageRank).
* **Concurrency:** Utilizes multi-threading (`ThreadPoolExecutor`) to fetch posting lists from the bucket in parallel, optimizing I/O operations.
* **Cloud Architecture:** Indexes are stored in a GCS bucket (`raz439`), while lightweight metadata is loaded into RAM for fast access.

## 📂 Project Structure

The repository contains the following key files:

* **`search_frontend.py`**: The main Flask application. It initializes the search engine, loads the necessary indices into memory (inverted index, document norms, PageRank), and exposes the HTTP endpoints (`/search`, `/search_body`, etc.). It contains the core ranking logic.
* **`inverted_index_gcp.py`**: A utility class for handling the Inverted Index. It includes the `MultiFileReader` class, which is responsible for reading binary posting lists directly from the GCS bucket or local disk.
* **`startup_script_gcp.sh`**: A shell script used to initialize the GCP Compute Engine instance, install dependencies, and download necessary code/data upon startup.
* **`run_frontend_in_gcp.sh`**: Commands and instructions for deploying the engine to GCP (reserving IP, creating the instance).
* **`run_frontend_in_colab.ipynb`**: A Jupyter Notebook used for development, testing the engine logic, and running the `check_quality` script locally before deployment.
* **`queries_train.json`**: A dataset containing training queries and their ground-truth relevant documents for evaluation.

## 🛠️ Prerequisites & Installation

To run this project locally, you need Python 3.8+ and the following libraries:

```bash
pip install flask google-cloud-storage pandas numpy scikit-learn
