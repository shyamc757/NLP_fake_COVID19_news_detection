import nltk

def download_nltk_libraries():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')

if __name__ == "__main__":
    download_nltk_libraries()