import numpy as np 
import math
from collections import defaultdict
from text_processing import strip_text

### misc ###

def constant(matrices_list):
    """
    Desc: Concatenates matrices together and adjoins a column of 1s
    """

    n_rows = len(matrices_list[0])
    ones = np.ones(shape=(n_rows, 1))
    return np.concatenate(matrices_list + [ones], axis=1)

### bag of words features ###

def feat_cnt(text, bag, gram_id, n, no_punct=True, no_stopwords=True, stem=False):
    """
    text = string
    bag = set of grams
    gram_id = dictinary mapping a gram to an integer
    n = length of gram

    Build bag of words feature from n-grams. Default is to remove punctuation
    """

    feat = [0] * len(gram_id)
    grams = strip_text(text, n, no_punct, no_stopwords, stem)
    for g in grams:
        if g in bag: feat[gram_id[g]] += 1

    return feat

### tfidf features ###

def tf(w, d):
    """
    w = an n-gram
    d = list of n-grams

    Desc: computes the tf of w in d
    """

    return d.count(w)

def idf(w, D, base):
    """
    w = n-gram
    D = list of documents, where each document d is a list of n-grams
    base = base of the logarithm

    Desc: computes the idf value of w in D
    """

    num = len(D)
    den = len([d for d in D if w in d])

    if num > 0 and den > 0:
        return math.log(num / den, base)
    else:
        return 0

def tfidf(w, d, D, base):
    """
    w = n-gram
    d = list of n-grams -- the review text
    D = list of documents d
    base = base of logarithm used to compute ifd

    Desc: computes tfidf value
    """

    return tf(w, d) * idf(w, D, base)

def get_gram_idf(bag, D, base):
    """
    bag = set of n-grams
    D = list of documents, where each document d is a list of n-grams
    base = base of logarithm

    Desc: Computes idf values of each n-gram in bag wrt to the set of documents D. 
          Return dictionary
    """

    gram_idf = defaultdict(float)
    for g in bag:
        gram_idf[g] = idf(g, D, base)

    return gram_idf

def feat_tfidf(text, bag, gram_id, gram_idf, n, no_punct=True, no_stopwords=True, stem=False):
    """
    text = string
    bag = set of n-grams
    gram_id = dictionary mapping an n-gram to integer
    gram_idf = dictionary mapping an n-gram to its idf value 
    n = length of gram

    Desc: Build tfidf feature from grams
    """

    feat = [0] * len(gram_id)
    grams = strip_text(text, n, no_punct, no_stopwords, stem)

    for g in grams:
        if g in bag: feat[gram_id[g]] = tf(g, grams) * gram_idf[g]

    return feat

### classifying words as positive/negative ####

def classify_pos_neg(reviews, n, no_punct=True, no_stopwords=True, stem=False):
    """
    reviews = list of reviews, each review is a dictionary containing keys {funny,posted,...}
    n = length of gram

    Desc: Go through each review and classify every gram according to the label recommendation.
          That is, if a review NOT a recommendation, classify every gram as negative
          After processing every review, returns a dictionary saying how positive/negative
          a word is. -1 = very negative, 0 = neutral, 1 = very positive
    """

    gram_cnt = defaultdict(int)
    gram_sentiment_cnt = defaultdict(int)
    for r in reviews:
        label = 1 if r["recommend"] == True else -1
        grams = strip_text(r["review"], n, no_punct, no_stopwords, stem)
        for g in grams:
            gram_cnt[g] += 1
            gram_sentiment_cnt[g] += label

    ratings = defaultdict(float)
    for g in gram_cnt.keys():
        ratings[g] = gram_sentiment_cnt[g] / gram_cnt[g]

    return ratings

def feat_rating(text, n, gram_rating, no_punct=True, no_stopwords=True, stem=False):
    """
    text =  string of words
    bag = set of grams
    gram_ratings = output of classify_pos_neg
    """

    rating = 0
    grams = strip_text(text, n, no_punct, no_stopwords, stem)
    for g in grams:
        rating += gram_rating[g]
    
    return [rating / len(grams) if len(grams) > 0 else 0]
