from text_processing import readJSON, common_grams, bag, strip_text
from collections import defaultdict
from metrics import measure

# Each entry is of the format [user_id, user_url, reviews]
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

# Only consider the top 1000 unigrams, removing punct + stopwords
n = 1
k = 1000
top_grams = common_grams([r["review"] for r in train_data], n, k)
grams = bag(top_grams)

# Calculating probabilities
# prob(label = positive) = total num positive reviews / total reviews
total_reviews = len(train_data)
positive_reviews = [r["review"] for r in train_data if r["recommend"] == True]
p_positive = len(positive_reviews) / total_reviews
p_negative = 1 - p_positive

# prob(gram is positive) = total num positive grams / total grams
gram_cnt = {g : cnt for (cnt, g) in top_grams}
positive_gram_cnt = defaultdict(int)
for r in positive_reviews:
    r_grams = strip_text(r, n)
    for g in grams:
        positive_gram_cnt[g] += r_grams.count(g)
    
gram_positive_prob = {g : positive_gram_cnt[g] / gram_cnt[g] for g in grams}

# Testing model on test data

def prod(S):
    product = 1
    for x in S:
        product *= x
    return product

y_test = [1 if r["recommend"] == True else 0 for r in test_data]
y_pred = []

for r in test_data:
    r_grams = strip_text(r["review"], n)
    filtered = [g for g in r_grams if g in grams]

    # Guess yes if there are no top_grams appearing in text
    if len(filtered) == 0:
        y_pred.append(1)
    # Compute according to naive probability model
    else:
        prob_label_positive = p_positive * prod([gram_positive_prob[g] for g in filtered])
        prob_label_negative = p_negative * prod([1-gram_positive_prob[g] for g in filtered])

        if prob_label_positive > prob_label_negative:
            y_pred.append(1)
        else:
            y_pred.append(1)

print("Baseline: (acc, TP, TN, FP, FN, BER) = {}".format(measure(y_test, y_pred, 5)))
