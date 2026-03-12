import task2
import csv
import math 
import pandas as pd
from collections import defaultdict
TEST_QUERIES = "test-queries.tsv"
OUTPUT_FILE = "tfidf.csv"
INVERTED_INDEX = task2.build_inverted_index() # word : [(pid, tf_t)]

def calculate_big_n(document):
    # calculate N, the number of docs in the collection
    big_n = 0
    with open(document, newline='') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        big_n = len(list(tsv_reader))
    return big_n

def calculate_passage_tfidf():
    BIG_N = calculate_big_n(task2.COLLECTION)
    inverted_index = INVERTED_INDEX  # word : [(pid, tf_t)]
    passage_tfidf_vectors = {}  # pid : {term: tfidf}

    for term, postings in inverted_index.items():
        idf_t = math.log(BIG_N / len(postings), 10)
        #print('postings = ', postings)
        for pid, tf_t in postings.items():
            tf_idf = tf_t * idf_t
            if pid not in passage_tfidf_vectors:
                passage_tfidf_vectors[pid] = {}
            passage_tfidf_vectors[pid][term] = tf_idf

    return passage_tfidf_vectors

def calculate_query_tfidf():
    BIG_N = calculate_big_n(task2.COLLECTION)  # use passage corpus N
    test_queries = pd.read_csv(TEST_QUERIES, sep='\t', header=None)
    inverted_index = INVERTED_INDEX
    query_tfidf_vectors = {}  # qid : {term: tfidf}

    for _, row in test_queries.iterrows():
        qid = row[0]
        query = row[1].split()

        # compute query term frequencies first
        query_tf = {}
        for term in query:
            query_tf[term] = query_tf.get(term, 0) + 1

        # then compute tfidf using passage IDF
        query_tfidf_vectors[qid] = {}
        for term, tf_q in query_tf.items():
            if term in inverted_index:
                idf_t = math.log(BIG_N / len(inverted_index[term]), 10)
                query_tfidf_vectors[qid][term] = tf_q * idf_t

    return query_tfidf_vectors

def calculate_cosine_similarity():
    # Calculate the cosine similarity between the TF-IDF vectors of the queries and the passages
    query_tfidf_vectors = calculate_query_tfidf()
    print("finished calculating query tfidf vectors")
    passage_tfidf_vectors = calculate_passage_tfidf()
    print("finished calculating passage tfidf vectors")
    cosine_similarity_scores = {} # (qid, pid) : cosine similarity score
    for qid, query_vector in query_tfidf_vectors.items():
        for pid, passage_vector in passage_tfidf_vectors.items():
            # calculate cosine similarity between query_vector and passage_vector
            inner_product = sum(query_vector[term] * passage_vector.get(term, 0) for term in query_vector)
            query_length = math.sqrt(sum(value ** 2 for value in query_vector.values()))
            #print("query_length = ", query_length)
            passage_length = math.sqrt(sum(value ** 2 for value in passage_vector.values()))
            #print("passage_length = ", passage_length)
            if query_length > 0 and passage_length > 0:
                cosine_similarity = inner_product / (query_length * passage_length)
                cosine_similarity_scores[(qid, pid)] = cosine_similarity

    return cosine_similarity_scores


if __name__ == "__main__":
    cosine_scores = calculate_cosine_similarity()



        


        
