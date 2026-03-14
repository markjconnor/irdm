import task1, task2
import csv
import math 
import pandas as pd
from collections import defaultdict
from task2 import build_inverted_index
TEST_QUERIES = "test-queries.tsv"
TFIDF_OUTPUT_FILE = "tfidf.csv"
INVERTED_INDEX = build_inverted_index(task2.COLLECTION) # word : [(pid, tf_t)]

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

def calculate_cosine_similarity(big_n, test_queries, passage_tfidf_vectors, query_tfidf_vectors, candidate_passages):
    # Calculate the cosine similarity between the tf-idf vectors of the queries and the passages

    candidates = calculate_query_candidates(candidate_passages)
    cosine_scores = {} # (qid, pid) : cosine similarity score
    for qid, query_vector in query_tfidf_vectors.items():

        candidate_pids = candidates.get(qid, set())
        for pid in candidate_pids:
            if pid not in passage_tfidf_vectors:
                continue
            passage_vector = passage_tfidf_vectors[pid]
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

def calculate_query_candidates(candidate_passages):
    
    candidates = defaultdict(set) # qid : set of candidate pids
    for _, row in candidate_passages.iterrows():
        qid, pid = row[0], str(row[1]) # ensure pid is a string for consistent key types
        candidates[qid].add(pid)
    return candidates

def output_results(table, test_queries):

    output = []
    qids = test_queries["qid"].tolist()
    for qid in qids:
        # get all passages and scores for this query
        passage_scores = {pid: score for (q, pid), score in table.items() if q == qid}
        # sort by score and take top 100
        top_passages = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)[:100]
        for pid, score in top_passages:
            output.append({"qid": qid, "pid": pid, "score": score})

    output_df = pd.DataFrame(output)
    return output_df

def calculate_bm25(candidate_passages, k1=1.5, k2=100, b=0.75):
    query_candidates = calculate_query_candidates(candidate_passages)
    
   

    #K = k1 * ((1 - b) + b * (document_length[pid] / average_document_length))

    n1 = {qid: {} for qid in query_candidates.keys()}
    n2 = {qid: {} for qid in query_candidates.keys()}
    bm25 = {qid: {} for qid in query_candidates.keys()}

    passages_tf = {}

    for term, tfs in INVERTED_INDEX.items():
        for pid, tf in tfs.items():
            passages_tf[pid] = {}
            
    for term, tfs in INVERTED_INDEX.items():
        for pid, tf in tfs.items(): 
            passages_tf[pid][term] = passages_tf[pid].get(term,0) + INVERTED_INDEX[term][pid]

    queries_tf = {qid: {} for qid in test_queries["qid"].tolist()}
    for qid, query_terms in zip(test_queries["qid"].tolist(), task1.get_vocabulary(task1.DATASET, None)):
        term_freq = {}
        for term in query_terms:
            term_freq[term] = term_freq.get(term, 0) + 1
        queries_tf[qid] = term_freq
    

    for qid, term_frequency in queries_tf.items():
        for term in term_frequency.keys():
            if term in INVERTED_INDEX.keys():
                n1[qid][term] = len(INVERTED_INDEX[term])
            else:
                n1[qid][term] = 0

    for qid, tf in queries_tf.items():
        terms = tf.keys()
        for pid, tf_t in passages_tf.items():
            n2[qid][pid] = {}
            for term in terms:
                if term in tf_t.keys():
                    n2[qid][pid][term] = passages_tf[pid][term]
    
    document_length = {}
    for pid, row in passages_tf.items():
        document_length[pid] = sum(row.values())

    average_document_length = sum(document_length.values())/len(document_length)

    for qid, passages in n2.items():
        for pid, terms in passages.items():
            score = 0
            for term, f_i in terms.items():
                n_i = n1[qid][term]
                qf_i = queries_tf[qid][term]
                K = k1 * ((1 - b) + b * (document_length[pid] / average_document_length))
                term_score = math.log( ((len(passages_tf) - n_i + 0.5) / (n_i + 0.5)) * ((k1 + 1) * f_i / (K + f_i)) * ((k2 + 1) * queries_tf[qid][term] / (k2 + queries_tf[qid][term])))
                score += term_score
            bm25[qid][pid] = score

    return bm25



if __name__ == "__main__":
    candidate_passages = pd.read_csv(task2.COLLECTION, sep='\t', header=None)
    test_queries = pd.read_csv(TEST_QUERIES, sep='\t', header=None)
    test_queries.columns = ["qid","text"]
    big_n = calculate_big_n(task2.COLLECTION)

    passage_tfidf = calculate_passage_tfidf(big_n)
    query_tfidf = calculate_query_tfidf(big_n, test_queries)

    cosine_scores = calculate_cosine_similarity(big_n, test_queries, passage_tfidf, query_tfidf, candidate_passages)
    tf_idfs = output_results(cosine_scores, test_queries)
    tf_idfs.to_csv(TFIDF_OUTPUT_FILE, index=False, header=False) # no headers

    bm25 = calculate_bm25(candidate_passages)
    print(bm25)
    bm25_scores = output_results(bm25, test_queries)
    bm25_scores.to_csv("bm25.csv", index=False, header=False)




        


        
