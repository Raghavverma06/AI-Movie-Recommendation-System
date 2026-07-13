**Intern ID:** CITS2709
# 🎬 AI Movie Recommendation System

A **production-grade, fully offline Hybrid Movie Recommendation Platform** built using **Python, Streamlit, and Scikit-Learn**.

This project combines **Content-Based Filtering**, **Collaborative Filtering**, and a **Hybrid Recommendation Engine** to generate intelligent movie recommendations without relying on any external APIs.

---

# 📖 Project Overview

The **AI Movie Recommendation System** is an advanced recommendation platform developed as part of an **AI & Machine Learning Internship**.

Unlike traditional recommendation systems that depend on cloud services, this application operates **100% offline**, requiring **no external APIs** such as TMDB, OMDb, or OpenAI.

During the first launch, the system automatically checks whether the **MovieLens dataset** is available. If it is not found, the application generates a highly realistic synthetic dataset consisting of:

- 🎬 **1,000 Movies**
- 👤 **600 Users**
- ⭐ **100,000 Ratings**
- 📝 Rich metadata including:
  - Genres
  - Descriptions
  - Directors
  - Cast
  - Keywords

The recommendation engine combines multiple recommendation strategies to provide highly relevant and personalized movie suggestions.

---

# 🚀 Recommendation Algorithms

The platform integrates three complementary recommendation techniques.

## 1️⃣ Content-Based Filtering

Uses:

- TF-IDF Vectorization
- Cosine Similarity

Movie recommendations are generated based on similarities between:

- Genres
- Directors
- Cast
- Keywords
- Movie Description

---

## 2️⃣ Collaborative Filtering

Uses:

- User-Item Rating Matrix
- Nearest Neighbors
- Truncated SVD Matrix Factorization

Recommendations are generated using historical user behavior and latent preference learning.

---

## 3️⃣ Hybrid Recommendation Engine

The final recommendation score is computed by combining:

- Content Similarity
- Rating Similarity
- Personalized Predicted Ratings

This produces significantly better recommendations than using any single recommendation technique alone.

---

# 🛠 Tech Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python 3.9+ |
| User Interface | Streamlit |
| Machine Learning | Scikit-Learn |
| Data Processing | Pandas, NumPy |
| Data Visualization | Plotly |
| Model Serialization | Joblib |
| System Monitoring | Psutil |
| Version Control | Git & GitHub |

---

# 🏗 System Architecture

```
                  +-----------------------------------+
                  |           User Input              |
                  | (Query Title + Optional User ID)  |
                  +-----------------+-----------------+
                                    |
                                    v
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
|  Content-Based Model  |                       | Collaborative Model   |
| TF-IDF + Cosine Sim.  |                       | User Rating Matrix    |
+-----------+-----------+                       +-----------+-----------+
            |                                               |
            |                                               |
            |                                   +-----------+-----------+
            |                                   | Nearest Neighbors     |
            |                                   +-----------+-----------+
            |                                               |
            |                       +-----------------------+
            |                       |
            |                       v
            |           +-----------------------+
            |           | Matrix Factorization  |
            |           | Truncated SVD         |
            |           +-----------+-----------+
            |                       |
            +-----------+-----------+
                        |
                        v
         +-------------------------------------+
         | Hybrid Recommendation Engine        |
         |                                     |
         | Score =                             |
         | w1 × Content Similarity             |
         | + w2 × Rating Similarity            |
         | + w3 × Predicted Rating             |
         +------------------+------------------+
                            |
                            v
               Top Movie Recommendations
```

---

# 📂 Project Directory Structure

```
AI Movie Recommendation System/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── assets/
├── data/
├── model/
├── outputs/
├── reports/
├── screenshots/
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── train.py
    ├── recommend.py
    ├── hybrid.py
    └── visualization.py
```

---

# 🚀 Installation Guide

## 1. Clone the Repository

```bash
git clone https://github.com/Raghavverma06/AI-Movie-Recommendation-System.git

cd AI-Movie-Recommendation-System
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 4. Train the Recommendation Models

This command:

- Generates synthetic data (if necessary)
- Trains all recommendation models
- Computes evaluation metrics
- Saves trained models into the **model/** directory

```bash
python -m src.train
```

---

## 5. Launch the Streamlit Dashboard

```bash
python -m streamlit run app.py
```

Open your browser and visit

```
http://localhost:8501
```

---

# 🧠 Recommendation Pipeline

## 1. Metadata Preparation

The preprocessing pipeline extracts:

- Genres
- Director
- Cast
- Keywords
- Description

These fields are merged into a single feature representation called the **Movie Soup**.

---

## 2. Text Vectorization

The movie metadata is transformed using

- TF-IDF Vectorizer
- 1–2 Gram Features

A Cosine Similarity matrix is then generated.

---

## 3. Collaborative Learning

A User-Item Rating Matrix is created.

Two collaborative models are trained:

- Nearest Neighbors
- Truncated SVD Matrix Factorization

These models learn user preferences and hidden relationships between movies.

---

## 4. Hybrid Recommendation

The final recommendation score combines

- Content Similarity
- Rating Similarity
- Personalized Rating Prediction

using weighted ensemble learning.

---

## 5. Confidence Estimation

Each recommendation includes a confidence score calculated from

- Similarity agreement
- Rating consistency
- User preference alignment

---

# 🖥 Dashboard Features

The Streamlit dashboard contains multiple professional modules.

### 🏠 Home

- Dataset Overview
- KPI Cards
- Algorithm Accuracy
- Project Summary

---

### 🎬 Recommend Movies

- Movie Search
- Adjustable Recommendation Weights
- Personalized Suggestions
- Export Recommendations
- Explainable AI ("Why this recommendation?")

---

### 📊 Analytics

Interactive Plotly Charts

- Rating Distribution
- Genre Distribution
- Movie Counts
- Dataset Insights

---

### 📈 Genre Statistics

- Top Genres
- Average Ratings
- Genre Performance

---

### ⭐ Top Rated Movies

Displays the highest-rated movies based on average ratings and minimum vote thresholds.

---

### 🔥 Popular Movies

Lists movies with the highest user engagement.

---

### 🧠 Recommendation Insights

Displays

- Similarity Heatmaps
- Recommendation Confidence
- Feature Analysis

---

### 📊 Model Comparison

Compares

- RMSE
- MAE
- Coverage
- Prediction Latency

for

- Content-Based Filtering
- Collaborative Filtering
- Hybrid Recommendation Engine

---

### 📂 Dataset Explorer

Search, filter and explore the movie dataset.

---

### 📜 Recommendation History

Tracks

- Search History
- Generated Recommendations
- Exportable CSV Logs

---

### ⚙ System Performance

Displays

- CPU Usage
- RAM Usage
- Model Loading Time
- Inference Latency

---

### 👨‍💻 About Developer

Professional developer profile and project information.

---

# ⭐ Key Features

- ✅ Fully Offline Recommendation System
- ✅ Hybrid Recommendation Engine
- ✅ Content-Based Filtering
- ✅ Collaborative Filtering
- ✅ Automatic Synthetic Dataset Generation
- ✅ Interactive Streamlit Dashboard
- ✅ Modern Glassmorphism UI
- ✅ Plotly Interactive Visualizations
- ✅ Recommendation History Tracking
- ✅ Model Comparison Dashboard
- ✅ Dataset Explorer
- ✅ CSV Export Support
- ✅ Modular Python Architecture
- ✅ Production-Ready Project Structure

---

# 👨‍💻 Developer

**Name:** Raghav Verma

**Program:** B.Tech Computer Science Engineering (Artificial Intelligence & Machine Learning)

**Institution:** Manav Rachna International Institute of Research and Studies (MRIIRS)

**Intern ID:** CITS2709

---

# 📄 License

This project is developed for educational and internship purposes.

---

⭐ If you found this project helpful, consider giving it a **Star** on GitHub.