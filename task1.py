import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

DATASET="passage-collection.txt"

def get_vocabulary(dataset, stop_words):
    
    with open(dataset) as f:
        text = f.read()
        # TODO: In report, TEXT PRE-PROCESSING CHOICE: Chose to ignore case sensitivity
        text = text.lower()
        words = text.split()
        if stop_words:
            words = [word for word in words if word not in stop_words]
    vocabulary = set(words)

    print("Size of the vocabulary = ", len(vocabulary)) 
    return vocabulary

def count_term_frequency(dataset, stop_words):
  with open(dataset) as f:
    text = f.read()
    words = text.split()
    if stop_words:
        words = [word for word in words if word not in stop_words]

  term_frequency = {}
  for word in words:
    term_frequency[word] = term_frequency.get(word, 0) + 1

  return term_frequency

def plot_figure_1(df):
   # Figure 1: plot normalised frequency against frequency ranking
    plt.figure(figsize=(10, 6))
    plt.plot(df["Rank"], df["Normalised Frequency"], marker='o')
    plt.xlabel('Rank')
    plt.ylabel('Normalised Frequency')
    plt.title('Term Frequency vs Rank')
    plt.grid(True)
    plt.savefig("figure_1.png")

def plot_figure_2(df):
   # Figure 2: create a log log plot of Rank*Frequency vs Zipf's Law Distribution
    plt.figure(figsize=(10, 6))
    plt.loglog(df["Rank"], df["Normalised Frequency"], marker='o', label='Observed Data')
    plt.loglog(df["Rank"], df["Zipf's Law Distribution"], marker='x', label="Zipf's Law", linestyle='--')
    plt.xlabel('Rank')
    plt.ylabel('Normalised Frequency')
    plt.title("Zipf's Law: Term Frequency vs Rank")
    plt.legend()
    plt.grid(True)
    plt.savefig("figure_2.png")

def construct_df(tf):
    ranked_terms = sorted(tf.items(), key=lambda x: x[1], reverse=True)
    rows = []

    denominator = sum(1 / i for i in range(1, len(tf) + 1))

    for rank, (term, freq) in enumerate(ranked_terms, start=1):
        normalised_occurrence = (freq / len(tf)) / 100

        rows.append({
           "Word": term, 
            "Rank": rank,
           "Normalised Frequency": normalised_occurrence,
           "rank*frequency": rank * normalised_occurrence,
           "Zipf's Law Distribution": (1 / rank / denominator)
        })

    return pd.DataFrame(rows)

"""
How will the difference between the two distributions be affected if we also remove stop
words? Justify your answer by depicting (Figure 3) and quantifying this difference
"""
def plot_figure_3(df, df_stop_words_removed):
    plt.figure(figsize=(10, 6))
    plt.loglog(df["Rank"], df["Normalised Frequency"], marker='o', label='Observed Data')
    plt.loglog(df_stop_words_removed["Rank"], df_stop_words_removed["Normalised Frequency"], marker='o', label='Observed Data (Stop Words Removed)')
    plt.xlabel('Rank')
    plt.ylabel('Normalised Frequency')
    plt.title("Zipf's Law: Term Frequency vs Rank (Stop Words Removed)")
    plt.legend()
    plt.grid(True)
    plt.savefig("figure_3.png")

if __name__ == "__main__":
    vocabulary = get_vocabulary(DATASET, None)
    tf = count_term_frequency(DATASET, None)
    df = construct_df(tf)
    

    plot_figure_1(df)
    plot_figure_2(df)

    stop_words = stopwords.words('english')
    vocabulary2 = get_vocabulary(DATASET, stop_words)
    tf2 = count_term_frequency(DATASET, stop_words)
    df2 = construct_df(tf2)
    plot_figure_3(df, df2)

  