from load_data import data
import nltk
import cleantext
import emoji

def tokenize_text(df):
    # function to tokenize text in dataframe and return a list of tokens
    df['text'] = df['text'].apply(str)
    text_tokens = []
    for i in range(df.shape[0]):
        text = df['text'][i]
        tokens = nltk.word_tokenize(text)
        text_tokens.append(tokens)
    return text_tokens

def tag_text(df, text_tokens):
    # function that tags the tokens as part of speech and return list of tuples with word and the tag
    tagged_tokens = []
    for i in range(df.shape[0]):
        tagged = nltk.pos_tag(text_tokens[i])
        tagged_tokens.append(tagged)
    return tagged_tokens

def lemmatize_text(tagged_tokens):
    # function to lemmatize the tokens (word to root of its word, example, 'changing' to 'change')
    lemmatized_tokens = []
    for tagged in tagged_tokens:
        lemma_t = []
        for k,v in tagged:
            try:
                lemma = nltk.stem.WordNetLemmatizer().lemmatize(k, pos = v)
            except KeyError:
                lemma = k 
            lemma_t.append(lemma)
        lemmatized_tokens.append(lemma_t)
    return lemmatized_tokens

def remove_stopwords(lemmatized_tokens):
    # function to remove stopwords (words that do not add much to the context in terms of meaning)
    new_stopwords = set(nltk.corpus.stopwords.words('english'))
    lemmatized_no_stopwords = []

    for words in lemmatized_tokens:
        temp_list = []
        temp_list = [x for x in words if not x.lower() in new_stopwords]

        lemmatized_no_stopwords.append(temp_list)
    
    return lemmatized_no_stopwords

def clean_text(lemmatized_no_stopwords):
    # function to remove emojis, less than 2 lettered words, hyperlinks, numbers, punctuations, conversion to lowercase
    cleaned_list_of_list = []

    for row in lemmatized_no_stopwords:
        cleaned_list = []
        for token in row:

            # cleaning numbers, links, punctuations and converting to lowercase
            temp = cleantext.clean(token, 
                                reg=r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                                reg_replace="", lowercase=True, numbers=True, punct=True)

            # filtering out emojis
            temp_text = ""
            for char in temp:
                if not emoji.is_emoji(char):
                    temp_text = temp_text + char

            # taking in words with length >= 2 only
            if len(temp_text)>=2:
                cleaned_list.append(temp_text)

        cleaned_list_of_list.append(cleaned_list)
        
    # joining the tokens in sentences for every row
    for i in range(len(cleaned_list_of_list)):
        cleaned_list_of_list[i] = " ".join(cleaned_list_of_list[i])
    
    return cleaned_list_of_list



