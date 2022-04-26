import gzip
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict

def readJSON(path):
    for l in gzip.open(path, mode='r'):
        d = eval(l)
        yield d

def get_n_grams(words, n):
    """
    Desc: From a list of words, return a list of n-grams
    """

    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

def strip_text(text, n=1, no_punct=True, no_stopwords=True, stem=False):
    """
    text - is a string

    Desc: returns a list of n-grams from text. Default processing is:
            - remove punctuation
            - keep stop words
            - do not stem
          always lower-case words
    """

    # Text-processing phase
    t = text.lower()
    if no_punct:
        t = [c for c in t if c not in string.punctuation]
    else:
        t = [c if c not in string.punctuation else " " + c + " " for c in t]

    t = "".join(t)
    words = t.strip().split()
    if no_stopwords:
        sp = set(stopwords.words("english"))
        words = [w for w in words if w not in sp]
    if stem:
        ps = PorterStemmer()
        words = [ps.stem(w) for w in words]

    # Return list of n-grams
    return get_n_grams(words, n)

def common_grams(texts, n, k, no_punct=True, no_stopwords=True, stem=False):
    """
    texts = list of strings
    return value = list of tuples (count, n-gram)
    
    Desc: Processes data and returns a list of top k occurring n-grams. 
    By default, removes punctuation
    """
    
    n_gram_cnt = defaultdict(int)
    for t in texts:
        grams = strip_text(t, n, no_punct, no_stopwords, stem)
        for g in grams: n_gram_cnt[g] += 1

    cnts = [(n_gram_cnt[g], g) for g in n_gram_cnt]
    cnts.sort()
    cnts.reverse()
    if k == -1:
        return cnts
    else:
        return cnts[:k]

def bag(top_grams):
    """
    top_grams = list of (count, n-gram) tuples from text_common words

    Desc: returns a set of n-grams
    """
    return {g for (_,g) in top_grams}

def get_gram_id(bag):
    """
    bag = set of n-grams

    Desc: maps each n-gram to unique id
    """

    return dict(zip(bag, range(len(bag))))

def combine_top_grams(grams_list, k):
    """
    grams_list = list of lists, where grams_list[i] is the top i-grams, length k
    return val = list,where the ith entry is grams_list[i] containing only top grams
    
    Desc: sorts the grams across all n-grams and get top k frequency
    """
    
    top = []
    for grams in grams_list:
        top += grams

    top.sort()
    top.reverse()
    top = top[:k]

    filtered = []
    for grams in grams_list: 
        filtered.append([g for g in grams if g in top])

    return filtered

def write_grams(top_grams):
    """
    top_grams = list of tuples (count, gram)

    Desc: Writes into a file
    """

    n = len(top_grams[0].strip().split())
    gram_file = open("{}-grams.txt".format(n), "w")
    for g in top_grams:
       gram_file.write(str(g) + "\n")

    gram_file.close()

def get_n_tuples(coords, n):
    """
    Desc: From a list of objects, return a list of n-tuples
    """

    return [coords[i:i+n] for i in range(len(coords) - n + 1)]

def check_two_pattern(bigram, rules):
    """
    bigram = is a list of length 2 of tuples (word, POS)
    rules = list of rule, where each rule is a 2-tuple of sets X, Y

    Desc: if POS1 is in X and POS2 is in Y, accept  
    """
    x, y = bigram
    x_pos, y_pos = x[1], y[1]
    for rule in rules:
        if x_pos in rule[0] and y_pos in rule[1]:
            return 1
    return 0

def check_three_pattern(trigram, rules):
    """
    Similar to check_two_pattern, except a rule is of form X, Y, Z
    Accept if POS1 is in X, POS2 is in Y, POS3 is NOT in Z
    """
    x, y, z = trigram
    x_pos, y_pos, z_pos = x[1], y[1], z[1]
    for rule in rules:
        if x_pos in rule[0] and y_pos in rule[1] and z_pos not in rule[2]:
            return 1
    return 0

def common_bigrams_pos(texts, k, no_punct=True, no_stopwords=True, stem=False):
    """
    texts = list of strings
    return value = list of tuples (count, 2-gram) filted by parts of speech pattern
    
    Desc: Processes data and returns a list of top k occurring n-grams. 
    By default, removes punctuation
    """

    # Filter by parts of speech
    # two_patterns any bigrams that match the pair
    # three_patterns look at bigrams such that the third word is not in 3rd set, get first two
    two_patterns = [({"JJ"}, {"NN", "NNS"}), 
                    ({"RB", "RBR", "RBS"}, {"VB", "VBD", "VBN", "VBG"})]
    three_patterns = [({"RB", "RBR", "RBS"}, {"JJ"}, {"NN", "NNS"}),
                      ({"JJ"}, {"JJ"}, {"NN", "NNS"}),
                      ({"NN", "NNS"}, {"JJ"}, {"NN", "NNS"})]
    
    # Process text
    # Note: each gram here is of the form (Word, POS)
    gram_cnt = defaultdict(int)
    for t in texts:
        words = nltk.pos_tag(strip_text(t, 1, no_punct, no_stopwords, stem))
        bi_grams = get_n_tuples(words, 2)
        tri_grams = get_n_tuples(words, 3) 

        for bigram in bi_grams:
            if check_two_pattern(bigram, two_patterns):
                x, y = bigram
                gram = x[0] + " " + y[0]
                gram_cnt[gram] += 1

        for trigram in tri_grams:
            if check_three_pattern(trigram, three_patterns):
                x, y, z = trigram
                gram = x[0] + " " + y[0]
                gram_cnt[gram] += 1

    # Get most common bigrams
    cnts = [(gram_cnt[g], g) for g in gram_cnt]
    cnts.sort()
    cnts.reverse()
    if k == -1:
        return cnts
    else:
        return cnts[:k]
