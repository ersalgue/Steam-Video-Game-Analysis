from sklearn.linear_model import LogisticRegression
from text_processing import readJSON, common_grams, bag, get_gram_id, common_bigrams_pos
from features import feat_cnt, constant
from metrics import measure

# Each entry is of the format [user, user_url, reviews]
# Each review has keys [funny, posted, last_edited, item_id, helpful, recommend, review]
path = "./australian_user_reviews.json.gz"
data = []
for d in readJSON(path):
    data.append(d)

# Num reviews = 59305, roughly 80/20 train/test split
reviews = []
for user in data:
    for r in user["reviews"]:
        reviews.append(r)
train_data = reviews[0:47500]
test_data = reviews[47500:]

# Baseline bigrams
n = 2
train_texts = [d["review"] for d in train_data]
top_bigrams = common_grams(train_texts, n, 1000)
bi_bag = bag(top_bigrams)
bi_id = get_gram_id(bi_bag)
X = [feat_cnt(t, bi_bag, bi_id, n) for t in train_texts]
X = constant([X])
y = [1 if d["recommend"] == True else 0 for d in train_data]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
model_b = LogisticRegression(max_iter=1000, class_weight="balanced")
model_b.fit(X, y)

test_texts = [d["review"] for d in test_data]
X_test = [feat_cnt(t, bi_bag, bi_id, n) for t in test_texts]
X_test = constant([X_test])
y_test = [1 if d["recommend"] == True else 0 for d in test_data]
y_pred = model.predict(X_test)
y_pred_b = model_b.predict(X_test)

print("Bigrams Unbalanced: (acc, TP, TN, FP, FN, BER) = {}".format(measure(y_test, y_pred, 5)))
print("Bigrams Balanced: (acc, TP, TN, FP, FN, BER) = {}".format(measure(y_test, y_pred_b, 5)))

# Bigrams with part of speech filtering
top_bigrams = common_bigrams_pos(train_texts, 1000)
bi_bag = bag(top_bigrams)
bi_id = get_gram_id(bi_bag)
X = [feat_cnt(t, bi_bag, bi_id, n) for t in train_texts]
X = constant([X])
y = [1 if d["recommend"] == True else 0 for d in train_data]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
model_b = LogisticRegression(max_iter=1000, class_weight="balanced")
model_b.fit(X, y)

test_texts = [d["review"] for d in test_data]
X_test = [feat_cnt(t, bi_bag, bi_id, n) for t in test_texts]
X_test = constant([X_test])
y_test = [1 if d["recommend"] == True else 0 for d in test_data]
y_pred = model.predict(X_test)
y_pred_b = model_b.predict(X_test)

print("Bigrams w/ POS Unbalanced: (acc, TP, TN, FP, FN, BER) = {}".format(measure(y_test, 
                                                                                  y_pred, 5)))
print("Bigrams w/ POS Balanced: (acc, TP, TN, FP, FN, BER) = {}".format(measure(y_test, 
                                                                                y_pred_b, 5)))
