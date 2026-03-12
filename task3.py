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

"""
def calculate_query_tfidf():
    BIG_N = calculate_big_n()
    inverted_index = task2.build_inverted_index() # word : [(pid, tf_t)]
    with open(TEST_QUERIES, newline='') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
            qid = row[0]
            print("qid = ", qid)
            query = row[1].split()
            
            query_tfidf_score = {} # pid : similarity score
            
            for term in query:
                if term in inverted_index:
                    idf_t = math.log((BIG_N / len(inverted_index[term])) , 10)
                    for (_, tf_t) in inverted_index[term]:
                        tf_idf = tf_t * idf_t
                        query_tfidf_score[term] = query_tfidf_score.get(term, 0) + tf_idf

            #
            #sorted_scores = sorted(query_tfidf_score.items(), key=lambda x: x[1], reverse=True)
            ##write the top 200 positive results to a file
            #    for pid, score in sorted_scores[:100]:
            #with open(f"tfidf.csv", "a") as f:
            #        f.write(f"{qid}\t{pid}\t{score}\n")
            #
"""

def calculate_passage_tfidf():
    # Extract the TF-IDF vector representations of the passages 
    # using the inverted index you have constructed
    BIG_N = calculate_big_n(task2.COLLECTION)
    candidate_passages = pd.read_csv(task2.COLLECTION, sep='\t', header=None)
    passage_tfidf_vectors = {} # pid : {term: tf-idf score}
    inverted_index = INVERTED_INDEX # word : [(pid, tf_t)]
    for _, row in candidate_passages.iterrows():
        pid = row[1]
        passage = row[3].lower().split() #TODO: think about if lower() is correct
        for term in passage:
            if term in inverted_index:
                idf_t = math.log((BIG_N / len(inverted_index[term])) , 10)
                for (_, tf_t) in inverted_index[term]:
                    tf_idf = tf_t * idf_t
                    if pid not in passage_tfidf_vectors:
                        passage_tfidf_vectors[pid] = {}
                    #passage_tfidf_vectors[pid][term] = passage_tfidf_vectors[pid].get(term, 0) + tf_idf
                    passage_tfidf_vectors[pid][term] = tf_idf
    
    return passage_tfidf_vectors
    
def calculate_query_tfidf():
    # Extract the TF-IDF vector representations of the queries 
    # using the inverted index you have constructed
    BIG_N = calculate_big_n(TEST_QUERIES)
    test_queries = pd.read_csv(TEST_QUERIES, sep='\t', header=None)
    query_tfidf_vectors = {} # qid : {term: tf-idf score}
    inverted_index = INVERTED_INDEX # word : [(pid, tf_t)]
    for _, row in test_queries.iterrows():
        qid = row[0]
        query = row[1].split()
        for term in query:
            if term in inverted_index:
                idf_t = math.log((BIG_N / len(inverted_index[term])) , 10)
                for (_, tf_t) in inverted_index[term]:
                    tf_idf = tf_t * idf_t
                    if qid not in query_tfidf_vectors:
                        query_tfidf_vectors[qid] = {}
                    #query_tfidf_vectors[qid][term] = query_tfidf_vectors[qid].get(term, 0) + tf_idf
                    query_tfidf_vectors[qid][term] = tf_idf

    return query_tfidf_vectors

def calculate_cosine_similarity():
    # Calculate the cosine similarity between the TF-IDF vectors of the queries and the passages
    query_tfidf_vectors = calculate_query_tfidf()
    passage_tfidf_vectors = calculate_passage_tfidf()
    cosine_similarity_scores = {} # (qid, pid) : cosine similarity score
    for qid, query_vector in query_tfidf_vectors.items():
        for pid, passage_vector in passage_tfidf_vectors.items():
            # calculate cosine similarity between query_vector and passage_vector
            inner_product = sum(query_vector[term] * passage_vector.get(term, 0) for term in query_vector)
            query_length = math.sqrt(sum(value ** 2 for value in query_vector.values()))
            passage_length = math.sqrt(sum(value ** 2 for value in passage_vector.values()))
            if query_length > 0 and passage_length > 0:
                cosine_similarity = inner_product / (query_length * passage_length)
                cosine_similarity_scores[(qid, pid)] = cosine_similarity

    return cosine_similarity_scores



if __name__ == "__main__":
    cosine_scores = calculate_cosine_similarity()
    # write the top 100 results to a file
    with open(OUTPUT_FILE, "w") as f:
        for (qid, pid), score in sorted(cosine_scores.items(), key=lambda x: x[1], reverse=True)[:100]:
            f.write(f"{qid}\t{pid}\t{score}\n")
        
        


        
