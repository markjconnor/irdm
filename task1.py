import matplotlib.pyplot as plt
import pandas as pd

DATASET="passage-collection.txt"

def get_vocabulary(dataset):
    with open(dataset) as f:
        text = f.read()
        words = text.split()

    vocabulary = set(words)
    #print("Report the size of the identified index of terms (vocabulary):", len(vocabulary))

"""
implement a function (or a block of source code) that 
counts the number of occurrences of terms in the provided data set, 
plot (Figure 1) their probability of occurrence (normalised frequency) 
against their frequency ranking, and qualitatively justify that 
these terms follow Zipf’s law."""
def count_term_frequency(dataset="passage-collection.txt"):
  with open(dataset) as f:
    text = f.read()
    words = text.split()

  term_frequency = {}
  for word in words:
    word = word.lower()
    term_frequency[word] = term_frequency.get(word, 0) + 1


  return term_frequency

def calculate_zipf_distribution(rank, vocabulary_size):
   return 1 / rank / sum(1 / r for r in range(1, vocabulary_size + 1))

if __name__ == "__main__":
    get_vocabulary(DATASET)
    tf = count_term_frequency(DATASET)

    ranked_terms = sorted(tf.items(), key=lambda x: x[1], reverse=True)
    rows = []


    for rank, (term, freq) in enumerate(ranked_terms, start=1):
        normalised_occurrence = (freq / len(tf)) / 100

        rows.append({
           "Word": term, 
            "Rank": rank,
           "Normalised Frequency": normalised_occurrence,
           "rank*frequency": rank * normalised_occurrence,
           "Zipf's Law Distribution": calculate_zipf_distribution(rank, len(tf))
        })

    df = pd.DataFrame(rows)
    print(df.head(10))

    # Figure 2: create a log log plot of Rank*Frequency vs Zipf's Law Distribution
    plt.figure(figsize=(10, 6))
    plt.loglog(df["Rank"], df["Normalised Frequency"], marker='o', label='Observed Data')
    plt.loglog(df["Rank"], df["Zipf's Law Distribution"], marker='x', label="Zipf's Law", linestyle='--')
    plt.xlabel('Rank')
    plt.ylabel('Normalised Frequency')
    plt.title("Zipf's Law: Term Frequency vs Rank")
    plt.legend()
    plt.grid(True)
    plt.show()