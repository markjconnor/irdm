import task2 
import task3
import pandas as pd
INVERTED_INDEX = task2.build_inverted_index(task2.COLLECTION) # word : [(pid, tf_t)]

def calculate_smoothing(candidate_passages, test_queries):
    # M_D = f_w_D / len(D)

    laplace = defaultdict(dict)
    dirichlet = defaultdict(dict)
    lidstone = defaultdict(dict)

    # uninverted index is needed -> passage_tf = pid : {word: tf}
    passage_tf = defaultdict(dict)
    for word, pid_count in INVERTED_INDEX.items():
        for pid, count in pid_count.items(): 
            passage_tf[pid][word] = passage_tf[pid].get(pid,0) + INVERTED_INDEX[word][pid]

    # for each word in each query, how many times does it appear in each document? (f_i)
    for qid, query_tf in qf_i.items():
        words = query_tf.keys()
        for pid, word_tf in passage_tf.items():
            f_i[qid][pid] = {}
            for word in words:
                if word in word_tf.keys():
                    f_i[qid][pid][word] = passage_tf[pid][word]

    # calculate |D|
    total_collection_length = 0
    collection_lengths = {}
    for word, passage_tf in INVERTED_INDEX.items():
        total_collection_length += sum(passage_tf.values())
        collection_length[word] = sum(passage_tf.values())

    qids = task3.calcaulte_query_candidates()
    V = len(INVERTED_INDEX) # number of unique words in the entire collection
    e = 0.1
    u = 50
    for qid, tfs in f_i.items():
        
        pids = qids[qid]
        for pid in pids:
            laplace[qid][pid] = 0
            lidstone[qid][pid] = 0
            dirichlet[qid][pid] = 0

            for word, f in tfs[pid].items():
                doc_length = sum(passage_tf[pid].values())

                laplace[qid][pid] += (f+1)/(D+V)
                lidstone[qid][pid] += (f+e)/(D+(e*V))
                
                dirichlet_value_1 = ( doc_length / (doc_length + u) * (f / doc_length) )    # (N / N + u ) * P(w|C_M)
                dirichlet_value_2 = ( u / (doc_length + u) * (collection_lengths[word]/total_collection_length))

                dirichlet[word] = math.log(dirichlet_value_1+dirichlet_value_2)

    return laplace, lidstone, dirichlet


if __name__ == "__main__":
    candidate_passages = pd.read_csv(task2.COLLECTION, sep='\t', header=None)
    candidate_passages.columns = ["qid", "pid", "query", "passage"]
    test_queries = pd.read_csv(task3.TEST_QUERIES, sep='\t', header=None)
    test_queries.columns = ["qid","text"]

    laplace, lidstone, dirichlet = calculate_smoothing()

    laplace_df = task3.output_results(laplace, test_queries)
    lidstone_df = task3.output_results(lidstone, test_queries)
    dirichlet_df = task3.output_results(dirichlet, test_queries)
    laplace_df.to_csv("laplace.csv", index=False, header=False) 
    lidstone_df.to_csv("lidstone.csv", index=False, header=False)
    dirichlet_df.to_csv("dirichlet.csv", index=False, header=False)  

