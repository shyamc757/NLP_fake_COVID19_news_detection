from load_data import data
from text_processing import tokenize_text, tag_text, lemmatize_text, remove_stopwords, clean_text
from count_vectorizer import return_count_embeddings
from tfidf_vectorizer import return_tfidf_embeddings
from modelling import return_classification_accuracy
import pandas as pd

def main():
    
    text_tokens = tokenize_text(data)
    tagged_tokens = tag_text(data, text_tokens)
    lemmatized_tokens = lemmatize_text(tagged_tokens)
    lemmatized_no_stopwords = remove_stopwords(lemmatized_tokens)
    cleaned_list_of_list = clean_text(lemmatized_no_stopwords)
    
    # adding the cleaned-text to the dataframe
    data['text_clean'] = cleaned_list_of_list
    
    indices = [(1,x) for x in range(1,4)]
    columns = ["count","tfidf"]

    accuracies_df = pd.DataFrame(columns=columns,index=pd.MultiIndex.from_tuples(indices)) # hierarchical indexing for tuple ngrams
    
    vectorizer_function_map = {
        "count" : return_count_embeddings,
        "tfidf" : return_tfidf_embeddings
    }
    
    for vectorizer_type in ["count","tfidf"]:
        for b in range(1,4): # the second part of the tuple for ngram range, the first part is always 1
            ngram = (1,b)
            X = vectorizer_function_map[vectorizer_type](data['text_clean'],ngram)
            y = data['label']
            
            accuracies_df.loc[ngram,vectorizer_type] = return_classification_accuracy(X,y)
    print(accuracies_df)

if __name__ == "__main__":
    main()
