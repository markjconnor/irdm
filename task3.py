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

def calculate_qf_i(test_queries):
    qf_i = {}  # qid : {term: tf_q}

    for _, row in test_queries.iterrows():
        qid = row["qid"]
        query = row["text"].split()

        # compute query term frequencies
        query_tf = {}
        for term in query:
            query_tf[term] = query_tf.get(term, 0) + 1
    
        qf_i[qid] = query_tf
    
    return qf_i

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
                idf_t = math.log(big_n / len(inverted_index[term]), 10)  # TODO: justify this version of calculating IDF -> different ones exist
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
        qid, pid = row["qid"], str(row["pid"]) # ensure pid is a string for consistent key types
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
    n_i = defaultdict(dict) # qid: {term: n_i}
    f_i = defaultdict(dict) # qid: {term: f_i}
    bm = defaultdict(dict)
    qf_i = calculate_qf_i(test_queries) 

    # for each word in each query, how many documents does it appear in? (n_i)
    for qid, query_tf in qf_i.items():
        for word, tf_q in query_tf.items():
            if word in INVERTED_INDEX:
                n_i[qid][word] = len(INVERTED_INDEX[word])
            else:
                n_i[qid][word] = 0

    # uninverted index is needed -> passage_tf = pid : {word: tf}
    passage_tf = defaultdict(dict)
    for word, tfs in INVERTED_INDEX.items():
        for pid, count in tfs.items(): 
            passage_tf[pid][word] = passage_tf[pid].get(pid,0) + INVERTED_INDEX[word][pid]

    # for each word in each query, how many times does it appear in each document? (f_i)
    for qid, query_tf in qf_i.items():
        words = query_tf.keys()
        for pid, word_tf in passage_tf.items():
            f_i[qid][pid] = {}
            for word in words:
                if word in word_tf.keys():
                    f_i[qid][pid][word] = passage_tf[pid][word]
    
    
    # calculate average document length
    document_length = {}
    for pid, row in passage_tf.items():
        document_length[pid] = sum(row.values())

    average_document_length = sum(document_length.values())/len(document_length)

    qids = calculate_query_candidates(candidate_passages)
    for qid, tfs in f_i.items():
        pids = qids[qid]
        for pid in pids:
            bm[qid][pid] = 0
            for word, f in tfs[pid].items():
                n = n_i[qid][word]
                qf = qf_i[qid][word]
                K = k1 * ((1 - b) + b * (document_length[pid] / average_document_length))

                idf = (len(passage_tf) - n + 0.5) / (n + 0.5)
                p_tf = ((k1 + 1) * f) / (K + f)
                q_tf = ((k2 + 1) * qf) / (k2 + qf)

                bm[qid][pid] += math.log(idf) * p_tf * q_tf
    
    #reformat in order to output results correctly
    flat_bm25 = {}
    for qid, pid_scores in bm.items():
        for pid, score in pid_scores.items():
            flat_bm25[(qid, pid)] = score
    return flat_bm25

    return flat_bm25
    

if __name__ == "__main__":
    candidate_passages = pd.read_csv(task2.COLLECTION, sep='\t', header=None)
    candidate_passages.columns = ["qid", "pid", "query", "passage"]
    test_queries = pd.read_csv(TEST_QUERIES, sep='\t', header=None)
    test_queries.columns = ["qid","text"]
    big_n = calculate_big_n(task2.COLLECTION)

    passage_tfidf = calculate_passage_tfidf(big_n)
    query_tfidf = calculate_query_tfidf(big_n, test_queries)

    cosine_scores = calculate_cosine_similarity(big_n, test_queries, passage_tfidf, query_tfidf, candidate_passages)
    tf_idfs = output_results(cosine_scores, test_queries)
    tf_idfs.to_csv(TFIDF_OUTPUT_FILE, index=False, header=False) # no headers

    bm25 = calculate_bm25(candidate_passages)
    bm25_scores = output_results(bm25, test_queries)
    bm25_scores.to_csv("bm25.csv", index=False, header=False)




        


        
