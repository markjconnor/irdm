import matplotlib.pyplot as plt
DATASET="passage-collection.txt"

def d1_1(dataset):
    with open(dataset) as f:
        text = f.read()
        words = text.split()

    vocabulary = set(words)
    print("Report the size of the identified index of terms (vocabulary):", len(vocabulary))

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

if __name__ == "__main__":
    d1_1(DATASET)

    tf = count_term_frequency(DATASET)


    ranked_terms = sorted(tf.items(), key=lambda x: x[1], reverse=True)

    ranks = []
    occurrence_probabilities = []

    for rank, (term, freq) in enumerate(ranked_terms, start=1):
        occurrence_probability = freq / len(tf)
        ranks.append(rank)
        occurrence_probabilities.append(occurrence_probability)

    plt.figure()
    plt.plot(ranks, occurrence_probabilities)
    plt.xlabel("Rank")
    plt.ylabel("Occurrence Probability")
    plt.title("Rank vs Occurrence Probability")
    plt.show()


       