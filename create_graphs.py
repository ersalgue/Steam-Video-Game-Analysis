from text_processing import readJSON, common_grams, common_bigrams_pos
import matplotlib.pyplot as plt

print("Reading data")
path = "./australian_user_reviews.json.gz"
data = []
for d in readJSON(path):
    data.append(d)

reviews = []
for user in data:
    for r in user["reviews"]:
        reviews.append(r["review"])

def plot_bar_top_gram(grams_set, title, subname):
    for n, grams in grams_set:
        x, y = [], []
        for freq, gram in grams:
            x.append(gram)
            y.append(freq)
    
        # Initial layout of bar chart, grams on y-axis, num_counts on x-axis
        x_pos = range(len(x))
        plt.barh(x, y, color="green")
        plt.xlabel("Count")
        plt.ylabel("Grams")
        plt.yticks(x_pos, x)
        plt.title(title)

        # Adds n-gram counts on top of the bars
        for i, val in enumerate(y):
            plt.text(val, i, " " + str(val), color="blue", va="center")

        file_name = "graphs/{}_top{}_{}_grams.svg".format(subname, k, n)
        plt.savefig(file_name, bbox_inches="tight")
        plt.show()

# Plot the frequencies of top k counts 
if False:
    n_grams = [1,2,3]
    title = "No Punctuation, Keep Stopwords, No Stemming"
    print("Getting grams: {}".format(title))
    k = 35
    grams_set = [(n, common_grams(reviews, n, k, True, False, False)) for n in n_grams]

    print("Plotting")
    plot_bar_top_gram(grams_set, title, "P--")

    title = "No Punctuation, No Stopwords, No Stemming"
    print("Getting grams: {}".format(title))
    grams_set = [(n, common_grams(reviews, n, k, True, True, False)) for n in n_grams]

    print("Plotting")
    plot_bar_top_gram(grams_set, title, "PS-")

    title = "No Punctuation, No Stopwords, Stemming"
    print("Getting grams: {}".format(title))
    grams_set = [(n, common_grams(reviews, n, k, True, True, True)) for n in n_grams]

    print("Plotting")
    plot_bar_top_gram(grams_set, title, "PSS")

# Plot bigrams chart with POS filtering
if True:
    k = 35
    title = "No Punctuation, No Stopwords, No Stemming: POS"
    print("Getting grams: {}".format(title))
    grams_set = [(2, common_bigrams_pos(reviews, k, True, True, False))]
    print("Plotting")
    plot_bar_top_gram(grams_set, title, "POS")
