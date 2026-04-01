import task2 
import task3
import pandas as pd
from collections import defaultdict
import math

INVERTED_INDEX = task2.build_inverted_index(task2.COLLECTION) # word : [(pid, tf_t)]

def calculate_smoothing(candidate_passages, test_queries):
    # M_D = f_w_D / len(D)

    laplace = defaultdict(dict)
    dirichlet = defaultdict(dict)
    lidstone = defaultdict(dict)

    # uninverted index is needed -> passage_tf = pid : {word: tf}
    passage_tf = defaultdict(dict)
    for word, pid_tfs in INVERTED_INDEX.items():
        for pid, _ in pid_tfs.items(): 
            passage_tf[pid][word] = passage_tf[pid].get(word,0) + INVERTED_INDEX[word][pid]


    # for each word in each query, how many times does it appear in each document? (f_i)
    qf_i = task3.calculate_qf_i(test_queries) 
    f_i = {}
    for qid, query_tf in qf_i.items():
        f_i[qid] = {}
        words = query_tf.keys()
        for pid, word_tf in passage_tf.items():
            f_i[qid][pid] = {}
            for word in words:
                if word in word_tf.keys():
                    f_i[qid][pid][word] = passage_tf[pid][word]

    # calculate |D|
    total_collection_length = 0
    collection_lengths = {}
    for word, tfs in INVERTED_INDEX.items():
        total_collection_length += sum(tfs.values())
        collection_lengths[word] = sum(tfs.values())

    candidate_qids = task3.calculate_query_candidates(candidate_passages)
    vocab_size = len(INVERTED_INDEX) # number of unique words in the entire collection
    e = 0.1
    u = 50
    for qid, tfs in f_i.items():
        qid = int(qid)
        candidate_pids = candidate_qids[qid]
        #print("sample candidate pid:", list(candidate_pids)[0])
        #print("sample f_i pid:", list(tfs.keys())[0])
        #print("overlap:", list(candidate_pids)[0] in tfs)
        #break
        for pid in candidate_pids:
            pid = int(pid)
            laplace[qid][pid] = 0
            lidstone[qid][pid] = 0
            dirichlet[qid][pid] = 0
            doc_length = sum(passage_tf[str(pid)].values())

            for word, qf in qf_i[qid].items():
                if word not in INVERTED_INDEX:
                    continue  # skip words not in vocabulary
                f = passage_tf[str(pid)].get(word, 0)

                laplace[qid][pid] += math.log((f + 1) / (doc_length + vocab_size))
                lidstone[qid][pid] += math.log((f + e) / (doc_length+(e * vocab_size)))
                
                dirichlet_value_1 = ( ( doc_length / (doc_length + u) ) * (f / doc_length) )    # (N / N + u ) * P(w|C_M)
                dirichlet_value_2 = ( u / (doc_length + u) * (collection_lengths[word]/total_collection_length))

                dirichlet[qid][pid] = math.log(dirichlet_value_1+dirichlet_value_2)


    #print('type(list(candidate_pids[0]))', type(list(candidate_pids)[0]))  # pid type in candidates
    #print('type(list(f_i[qid].keys())[0])', type(list(f_i[qid].keys())[0]))  # pid type in f_i
    #print('type(list(candidate_qids[0]))', type(list(candidate_qids)[0]))  # pid type in candidates
    #print('type(list(f_i.keys())[0])', type(list(f_i.keys())[0]))  # pid type in f_i

    return flatten(laplace), flatten(lidstone), flatten(dirichlet)

def flatten(dictionary):
    flat = {}
    for qid, pid_scores in dictionary.items():
        for pid, score in pid_scores.items():
            flat[(qid, pid)] = score
    return flat

if __name__ == "__main__":
    candidate_passages = pd.read_csv(task2.COLLECTION, sep='\t', header=None)
    candidate_passages.columns = ["qid", "pid", "query", "passage"]
    test_queries = pd.read_csv(task3.TEST_QUERIES, sep='\t', header=None)
    test_queries.columns = ["qid","text"]

    laplace, lidstone, dirichlet = calculate_smoothing(candidate_passages, test_queries)

    laplace_df = task3.output_results(laplace, test_queries)
    lidstone_df = task3.output_results(lidstone, test_queries)
    dirichlet_df = task3.output_results(dirichlet, test_queries)
    laplace_df.to_csv("laplace.csv", index=False, header=False) 
    lidstone_df.to_csv("lidstone.csv", index=False, header=False)
    dirichlet_df.to_csv("dirichlet.csv", index=False, header=False)  

