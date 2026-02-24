import task1
import csv
from collections import defaultdict
VOCAB_DATASET = task1.DATASET
COLLECTION = "candidate-passages-top1000.tsv"

def build_inverted_index():
    # removing stop words because stop words that appear in many passages are less informative for retrieval
    stop_words = task1.stopwords.words('english')
    vocab = task1.get_vocabulary(VOCAB_DATASET, stop_words)   

    """
    THOUGHT PROCESS 
    ****** 
    Your query is a word and you want to retrieve a passage from that query. 
    Store the words and the index and the passage id (pid) as the value
    ******
    """

    inverted_index = defaultdict(set)     # term : [pid]
    with open(COLLECTION, newline='') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
            pid = row[1]
            passage = row[3].lower().split()
            
            for word in passage:
                
                if word in vocab:
                    #print("word", word, " in vocab")
                    inverted_index[word].add(pid)
    
    return inverted_index

def query_inverted_index(query_term, inverted_index):
    if query_term in inverted_index:
        return inverted_index[query_term]


if __name__ == "__main__":
    inverted_index = build_inverted_index()

    print("inverted_index[\"tantalising\"] = ", inverted_index["tantalising"])
    query_term = "tantalising"
    result = query_inverted_index(query_term, inverted_index)
    print("length of results= ", len(result))
    print(result)





