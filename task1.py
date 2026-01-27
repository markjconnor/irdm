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
    if word in term_frequency:
      term_frequency[word] += 1
    else:
      term_frequency[word] = 1

  return term_frequency

def calculate_zipf_distribution(rank, vocabulary_size):
   return 1 / rank / sum(1 / r for r in range(1, vocabulary_size + 1))

if __name__ == "__main__":
    get_vocabulary(DATASET)
    tf = count_term_frequency(DATASET)

    ranked_terms = sorted(tf.items(), key=lambda x: x[1], reverse=True)
    rows = []

    LIMIT = 50
    for rank, (term, freq) in enumerate(ranked_terms[:LIMIT], start=1):
        normalised_occurrence = (freq / len(tf)) / 100

        rows.append({
           "Word": term, 
            "Rank": rank,
           "Normalised Frequency": normalised_occurrence,
           "rank*frequency": rank * normalised_occurrence,
           "Zipf's Law Distribution": calculate_zipf_distribution(rank, len(tf))
        })

    df = pd.DataFrame(rows)
    print(df)
"""
    RANK_LIMIT = 50
    plt.figure()
    plt.plot(ranks[0:RANK_LIMIT], occurrence_probabilities[0:RANK_LIMIT])
    plt.xlabel("Rank")
    plt.ylabel("Occurrence Probability")
    plt.title("Rank vs Occurrence Probability")
    plt.show()
"""

       