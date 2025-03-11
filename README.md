# Natural Language Processing for classification of news on Covid19 as 'fake' or 'true' 
An end-to-end project on classification of news (text data) on Covid19. Essential text processing steps including tokenization, tagging, lemmatization, cleaning (removal of stopwords, emojis, punctuations, hyperlinks, etc) followed by generation of embeddings using CountVectorizer and TfidfVectorizer. Lastly classification using Logistic Regression.  #NLP #Covid19 #LogisticRegression" 
___

## Instructions:
To run the project: 

### 1)  Install all the required packages
To run the project, install the required packages from **requirements-conda.txt** and **requirements-pip.txt** in your new conda environment by running the following commands in your terminal:
```bash
conda install --file requirements-conda.txt
pip install -r requirements-pip.txt
```

### 2) Download essential libraries for **nltk**
For downloading essential libraries for nltk, run the **nltk_setup.py** file using the following command:
```bash
python3 nltk_setup.py
```

### 3) Run the pipeline
Run the **main.py** file using the following command:
```bash
python3 main.py
```
This will run all the steps and give the accuracies for different ngram combinations as output for both CountVectorizer and TfidfVectorizer. You can use the following individual scripts independantly based on customized requirement of variables or functions contained in them:

#### i) Loading data
Run the **load_data.py** file. Simply loads the corona_fake.csv file into a pandas dataframe.

#### ii) Pre-processing text
Run the **text_processing.py** file. Contains functions to tokenize, tag, lemmatize, remove stopwords, clean (removal of numbers, emojis, punctuations, hyperlinks, etc) the text sent as argument.

#### iii) Vectorization
Run the **count_vectorizer.py** or **tfidf_vectorizer.py** for the numerical embeddings. Contain respective functions which take preprocessed textual data and ngram combination as inputs and return a vectorized form (embeddings).

#### iv) Modelling
Run the **modelling.py** file. Contains the Logistic Regression modelling code to return the accuracies after classification having taken numerical data as input. 
