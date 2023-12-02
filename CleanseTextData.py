'''
Alexandra DeGrandchamp, Final Project
Text Cleansing Module
'''
import pandas as pd

def isolate_text(filepath, text_field):
    '''
        Isolates specified text field from dataset, returns single column dataframe of that field and full data set
    '''
    file = pd.read_csv(filepath)
    text_only_dataset = pd.DataFrame(file[text_field])

    return text_only_dataset, file

def process_words(df):
    '''
        Master function for processing text data
        Removes numeric elements from text
        Tokenizes and stems using NLTK tools
        Detokenizes final text
        Returns both dataframe and list for use in tf-idf vectorizer
    '''
    #importing relevant nltk libraries
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    import nltk
    nltk.download('punkt')
    
    detokenizer = TreebankWordDetokenizer() #to remove individual lists in series
    
    copy = df.copy(deep=True)

    #function can have long processing time; printed statements provided for feedback
    copy['Item Full'] = copy['Item Full'].replace('\d+','',regex=True) #removes numeric data
    print('Numeric values removed')
    stemmed = copy.apply(lambda x: x.map(stem_text)) #both tokenizes and stems tokens
    print('Text stemmed')
    no_tokens = stemmed.apply(lambda x: x.map(detokenizer.detokenize)) #undoes tokenization for better processing
    print('Text detokenized')

    data_as_list = finalize_list(no_tokens)
    
    return no_tokens, data_as_list  

def stem_text(text):
    '''
        Helper function for process_words
        Tokenizes each string and then applies an English-language snowball stemmer to text
        Optimized for use as a mapped function
    '''
    from nltk.stem.snowball import SnowballStemmer
    from nltk.tokenize import word_tokenize

    stemmer = SnowballStemmer('english')
    return [stemmer.stem(word) for word in  word_tokenize(str(text))]

def finalize_list(df):
    '''
        Helper function for process_words
        Creates a list of strings to feed into tf-idf vectorizer
    '''
    df_list = df.values.tolist()
    final_list = [string for doc in df_list for string in doc]
    return final_list