import task2
import csv
import math 
import pandas as pd
from collections import defaultdict
TEST_QUERIES = "test-queries.tsv"
TFIDF_OUTPUT_FILE = "tfidf.csv"
INVERTED_INDEX = task2.build_inverted_index() # word : [(pid, tf_t)]

def calculate_big_n(document):
    # calculate N, the number of docs in the collection
    big_n = 0
    with open(document, newline='') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        big_n = len(list(tsv_reader))
    return big_n

def calculate_passage_tfidf(big_n):
    inverted_index = INVERTED_INDEX  # word : [(pid, tf_t)]
    passage_tfidf_vectors = {}  # pid : {term: tfidf}

    for term, passage_frequencies in inverted_index.items():
        idf_t = math.log(big_n / len(passage_frequencies), 10)
        for pid, tf_t in passage_frequencies.items():
            tf_idf = tf_t * idf_t
            if pid not in passage_tfidf_vectors:
                passage_tfidf_vectors[pid] = {}
            passage_tfidf_vectors[pid][term] = tf_idf

    return passage_tfidf_vectors

def calculate_query_tfidf(big_n, test_queries):
    inverted_index = INVERTED_INDEX
    query_tfidf_vectors = {}  # qid : {term: tfidf}

    for _, row in test_queries.iterrows():
        qid = row["qid"]
        query = row["text"].split()

        # compute query term frequencies
        query_tf = {}
        for term in query:
            query_tf[term] = query_tf.get(term, 0) + 1

        # compute tfidf using IDF representation from the corpus of the passages
        query_tfidf_vectors[qid] = {}
        for term, tf_q in query_tf.items():
            if term in inverted_index:
                idf_t = math.log(big_n / len(inverted_index[term]), 10)
                query_tfidf_vectors[qid][term] = tf_q * idf_t

    return query_tfidf_vectors

def calculate_cosine_similarity(big_n, test_queries):
    # Calculate the cosine similarity between the tf-idf vectors of the queries and the passages
    query_tfidf_vectors = calculate_query_tfidf(big_n, test_queries)
    passage_tfidf_vectors = calculate_passage_tfidf(big_n)
    cosine_scores = {} # (qid, pid) : cosine similarity score
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
                cosine_scores[(qid, pid)] = cosine_similarity

    return cosine_scores

def output_results(cosine_scores, test_queries):

    output = []
    qids = test_queries["qid"].tolist()
    for qid in qids:
        # get all passages and scores for this query
        passage_scores = {pid: score for (q, pid), score in cosine_scores.items() if q == qid}
        # sort by score and take top 100
        top_passages = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)[:100]
        for rank, (pid, score) in enumerate(top_passages, start=1):
            output.append({"qid": qid, "pid": pid, "rank": rank, "score": score}, ignore_index=True)

    output_df = pd.DataFrame(output)
    return output_df


if __name__ == "__main__":
    test_queries = pd.read_csv(TEST_QUERIES, sep='\t', header=None)
    test_queries.columns = ["qid","text"]
    big_n = calculate_big_n(task2.COLLECTION)

    cosine_scores = calculate_cosine_similarity(big_n, test_queries)
    tf_idfs = output_results(cosine_scores, test_queries)
    tf_idfs.to_csv(TFIDF_OUTPUT_FILE, index=False, header=False) # no headers



        


        
