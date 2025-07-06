# 🧪 Natural Language Processing for Classification of COVID-19 News as 'Fake' or 'True'

An end-to-end project on classification of COVID-19-related news (text data) as either *fake* or *true*. The workflow includes essential NLP steps like tokenization, POS tagging, lemmatization, and cleaning (removal of stopwords, emojis, punctuation, hyperlinks, etc.), followed by feature extraction using **CountVectorizer** and **TfidfVectorizer**, and classification using **Logistic Regression**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Instructions

### 1️⃣ Install Required Packages

To run the project, first install the required packages from **requirements-conda.txt** and **requirements-pip.txt** in your Conda environment:

```bash
conda install --file requirements-conda.txt
pip install -r requirements-pip.txt
```

---

### 2️⃣ Download NLTK Resources

Run the following command to download the necessary **NLTK** data files:

```bash
python nltk_setup.py
```

---

### 3️⃣ Run the Pipeline

Execute the entire workflow using:

```bash
python main.py
```

This will run all steps and display classification accuracies for different n-gram combinations using both CountVectorizer and TfidfVectorizer.

---

## 🧩 Module Descriptions

### 📄 i) Loading Data
`load_data.py`  
Loads the `corona_fake.csv` dataset into a Pandas DataFrame.

### 🧹 ii) Text Pre-processing
`text_processing.py`  
Functions for tokenization, tagging, lemmatization, and cleaning (removal of stopwords, emojis, numbers, punctuation, links).

### 🔢 iii) Vectorization
- `count_vectorizer.py`: Generates count-based embeddings  
- `tfidf_vectorizer.py`: Generates TF-IDF-based embeddings  
Both take cleaned text and n-gram config as input.

### 🤖 iv) Modeling
`modelling.py`  
Contains Logistic Regression classification pipeline to compute and return accuracy metrics from vectorized input.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to fork, star, or contribute to this repository!
