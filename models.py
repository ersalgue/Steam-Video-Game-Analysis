# %%
#https://pythonprogramming.net/sklearn-scikit-learn-nltk-tutorial/
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from text_processing import readJSON, common_grams, bag, get_gram_id
from features import feat_cnt, constant
from metrics import measureMetrics

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
train_sentiments = [d["recommend"] for d in train_data]

# %%
#print(train_texts[0])
#print(train_sentiments[0])

# %%
ngrams = 1
no_punct = True
no_stopwords = True
stem = False
numTopWords = 1000

# %%
top_grams = common_grams(train_texts, ngrams, numTopWords, no_punct,no_stopwords,stem)
gram_bag = bag(top_grams)
gram_id = get_gram_id(gram_bag)
X = [feat_cnt(t, gram_bag, gram_id, ngrams, no_punct,no_stopwords,stem) for t in train_texts]
X = constant([X])
y = [1 if d["recommend"] == True else 0 for d in train_data]

test_texts = [d["review"] for d in test_data]
X_test = [feat_cnt(t, gram_bag, gram_id, ngrams, no_punct,no_stopwords,stem) for t in test_texts]
X_test = constant([X_test])
y_test = [1 if d["recommend"] == True else 0 for d in test_data]

# %%
classWeight = "unbalanced"
classWeight2 = None

# %%
# training various models by passing in the sklearn models 
MNB_clf = MultinomialNB()
MNB_clf.fit(X,y)

y_pred = MNB_clf.predict(X_test)
print(classWeight +" MNB_clf")
measureMetrics(y_test,y_pred,5)

# %%
BernoulliNB_clf = BernoulliNB()
BernoulliNB_clf.fit(X,y)

y_pred = BernoulliNB_clf.predict(X_test)
print(classWeight +" BernoulliNB_clf")
measureMetrics(y_test,y_pred,5)

# %%
LogisticRegression_clf = LogisticRegression(max_iter=1000, class_weight=classWeight)
LogisticRegression_clf.fit(X,y)

y_pred = LogisticRegression_clf.predict(X_test)
print(classWeight +" LogisticRegression_clf")
measureMetrics(y_test,y_pred,5)

# %%
SGDClassifier_clf = SGDClassifier(class_weight=classWeight2)
SGDClassifier_clf.fit(X,y)

y_pred = SGDClassifier_clf.predict(X_test)
print(str(classWeight2) +" SGDClassifier_clf")
measureMetrics(y_test,y_pred,5)

# %%
LinearSVC_clf = LinearSVC(random_state=0,class_weight=classWeight2)
LinearSVC_clf.fit(X,y)

y_pred = LinearSVC_clf.predict(X_test)
print(str(classWeight2) +" LinearSVC_clf")
measureMetrics(y_test,y_pred,5)

# %%
""" too long to run
SVC_clf = SVC(class_weight=classWeight)
SVC_clf.fit(X,y)

y_pred = SVC_clf.predict(X_test)
print(classWeight +" SVC_clf")
measureMetrics(y_test,y_pred,5)

"""

# %%
""" too long to run
NuSVC_clf = NuSVC(nu=0.05,random_state=0,class_weight=classWeight)
NuSVC_clf.fit(X,y)

y_pred = NuSVC_clf.predict(X_test)
print(classWeight +" NuSVC_clf")
measureMetrics(y_test,y_pred,5)

"""
# %%
