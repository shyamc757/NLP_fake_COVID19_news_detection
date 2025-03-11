from load_data import data
from text_processing import tokenize_text, tag_text, lemmatize_text, remove_stopwords, clean_text
from sklearn.feature_extraction.text import CountVectorizer

def main():
    text_tokens = tokenize_text(data)
    tagged_tokens = tag_text(data, text_tokens)
    lemmatized_tokens = lemmatize_text(tagged_tokens)
    lemmatized_no_stopwords = remove_stopwords(lemmatized_tokens)
    cleaned_list_of_list = clean_text(lemmatized_no_stopwords)
    print(cleaned_list_of_list[0])
    
    

if __name__ == "__main__":
    main()



