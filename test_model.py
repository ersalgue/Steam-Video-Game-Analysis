# %%
# dictsize:10000 unigrams of top part of speech words: abdjectives, adverb, verbs
import nltk
import string, re
from features import feat_cnt, constant
from text_processing import readJSON, common_grams, bag, get_gram_id
from collections import defaultdict
from metrics import measureMetrics, measure
from sklearn.linear_model import LogisticRegression
# from analysis import gen_wordCloud

# %%
# Each entry is of the format [user, userId, reviewSet]
# Each review has keys [funny, posted, last_edited, item_id, helpful, recommend, review]
path = "./australian_user_reviews.json.gz"
data = []
for d in readJSON(path):
    data.append(d)

# %%
# Num reviews = 59305, roughly 80/20 train/test split
reviews = []
for user in data:
    for r in user["reviews"]:
        reviews.append(r)
train_data = reviews[0:47500]
test_data = reviews[47500:]
# %%
train_texts = [d["review"] for d in train_data]

# %%
topWords = common_grams(train_texts, 1, 10000,True,True)

# %%
topWrds = [w for n,w in topWords]
topWrds[:10]

# %%
nltk.download('averaged_perceptron_tagger')
POS = nltk.pos_tag(topWrds)

# %%
#  jj is adject, rb is adverb, and vb is verb
allowed_word_types = ["JJ","RB","VB"]
#allowed_word_types = ["NN"]
all_words = [w for w,p in POS if p in allowed_word_types]

# %%
len(all_words)

# gen_wordCloud(' '.join(all_words),"test_model")

# %%

bag_id = get_gram_id(all_words)
X = [feat_cnt(t, all_words, bag_id, 1) for t in train_texts]
X = constant([X])
y = [1 if d["recommend"] == True else 0 for d in train_data]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
model_b = LogisticRegression(max_iter=1000, class_weight="balanced")
model_b.fit(X, y)

test_texts = [d["review"] for d in test_data]
X_test = [feat_cnt(t, all_words, bag_id, 1) for t in test_texts]
X_test = constant([X_test])
y_test = [1 if d["recommend"] == True else 0 for d in test_data]
y_pred = model.predict(X_test)
y_pred_b = model_b.predict(X_test)

print("Unbalanced: (acc, TP, TN, FP, FN, BER) = {}".format(measure(y_test, 
                                                                                  y_pred, 5)))
print("Balanced: (acc, TP, TN, FP, FN, BER) = {}".format(measure(y_test, 
                                                                                y_pred_b, 5)))

# measureMetrics(y_test,y_pred,5)

# %%
