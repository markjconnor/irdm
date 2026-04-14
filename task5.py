import task2
import task3 
import gensim.downloader as api
from collections import defaultdict
import pandas as pd
import math
import nltk
import numpy as np
import re
MODEL_NAME = "glove-wiki-gigaword-100" # Previously: fasttext-wiki-news-subwords-300
TEST_QUERIES = task3.TEST_QUERIES

def calculate_average_precision(predicted_pids, ideal_qrels, k=25):
    num_relevant_found = 0
    precision_sum = 0.0
    
    total_relevant = sum(1 for rel in ideal_qrels.values() if rel[0] > 0)
    total_relevant = min(total_relevant, k)
    
    if total_relevant == 0:
        return 0.0
        
    for i, pid in enumerate(predicted_pids[:k]):
        rel = ideal_qrels.get(pid, [0])[0]
        if rel > 0:
            num_relevant_found += 1
            precision_sum += num_relevant_found / (i + 1)
            
    return precision_sum / total_relevant


def calculate_ndcg(predicted_pids, ideal_qrels, k=25):
    # Calculate DCG
    dcg = 0.0
    for i, pid in enumerate(predicted_pids[:k]):
        rel = ideal_qrels.get(pid, [0])[0]
        if rel > 0:
            dcg += rel / math.log2(i + 2) 
            
    # Ideal DCG (IDCG)
    idcg = 0.0
    ideal_rels = [rel[0] for rel in ideal_qrels.values()][:k]
    
    for i, rel in enumerate(ideal_rels):
        if rel > 0:
            idcg += rel / math.log2(i + 2)
            
    return dcg / idcg if idcg > 0 else 0.0

def optimise_bm25(candidate_passages, test_queries, train_data, inverted_index, doc_lengths, total_docs, avg_doc_length):

    # rank top 25 for relevant scores from train-data
    # calculate average precision
    # calculate NDCG
    train_rankings = get_train_rankings(train_data)
    query_train_data = train_data[['qid', 'queries']].drop_duplicates().rename(columns={'queries': 'text'})
    query_train_data['qid'] = query_train_data['qid'].astype(str)
    train_candidate_passages = {}
    for raw_qid, group in train_data.groupby('qid'):
        qid_str = str(raw_qid)
        train_candidate_passages[qid_str] = [str(pid) for pid in group['pid']]

    k1_range = [1.2, 1.6, 2.0]
    k2 = 100
    b_range = [0.5, 0.65, 0.8]
    
    best_ndcg = -1.0 
    best_params = None
    for k1 in k1_range:
        for b in b_range:
            bm25 = task3.calculate_bm25_fast(
                inverted_index=inverted_index,
                candidate_passages=train_candidate_passages, 
                queries_df=query_train_data, 
                doc_lengths=doc_lengths, 
                total_docs=total_docs, 
                avg_doc_length=avg_doc_length, 
                k1=k1, k2=k2, b=b
            )
            #print(f"--- DEBUG ---")
            #print(f"BM25 calculated results for these QIDs: {list(bm25.keys())[:3]}")
            #print(f"But the loop is looking for QIDs like: {list(train_rankings.keys())[:3]}")
            #print(f"-------------")
            total_ap = 0.0
            total_ndcg = 0.0
            num_queries = len(train_rankings)

            # Evaluate performance across all queries in the training set
            for qid, ideal_qrels in train_rankings.items():
                qid = str(qid)
                query_scores = bm25.get(qid, {})
                #print('query_scores=', query_scores) 

                predicted_pids = sorted(query_scores.keys(), key=lambda pid: query_scores[pid], reverse=True)
                
                safe_ideal_qrels = {str(p_id): rel for p_id, rel in ideal_qrels.items()}
                
                total_ap += calculate_average_precision(predicted_pids, safe_ideal_qrels, k=25)
                total_ndcg += calculate_ndcg(predicted_pids, safe_ideal_qrels, k=25)
                #print('total ndcg = ', total_ndcg)

            mean_ap = total_ap / num_queries
            mean_ndcg = total_ndcg / num_queries
            print('mean_ndcg=', mean_ndcg)
            print('mean_ap=', mean_ap)

            if mean_ndcg > best_ndcg:
                best_ndcg = mean_ndcg
                best_params = {"k1": k1, "k2": k2, "b": b}
    
    print(f"Best Parameters: k1={best_params['k1']}, b={best_params['b']} (NDCG@25: {best_ndcg:.4f})")
    
    return best_params

def evaluate_bm25(validation_data, inverted_index, doc_lengths, best_params, stop_words):
    
    val_total_docs = len(doc_lengths)
    val_avg_dl = sum(doc_lengths.values()) / val_total_docs if val_total_docs > 0 else 1.0
    
    val_queries_df = validation_data[['qid', 'queries']].drop_duplicates().rename(columns={'queries': 'text'})
    val_queries_df['qid'] = val_queries_df['qid'].astype(str)
    
    val_candidate_passages = {}
    for raw_qid, group in validation_data.groupby('qid'):
        qid_str = str(raw_qid)
        val_candidate_passages[qid_str] = [str(pid) for pid in group['pid']]
        
    val_rankings = get_train_rankings(validation_data)
    
    k1 = best_params['k1']
    b = best_params['b']
    k2 = best_params.get('k2', 100)
    
    print(f"\nEvaluating with optimal parameters: k1={k1}, b={b}")
    
    bm25_results = task3.calculate_bm25_fast(
        inverted_index=inverted_index,           
        candidate_passages=val_candidate_passages, 
        queries_df=val_queries_df, 
        doc_lengths=doc_lengths, 
        avg_doc_length=val_avg_dl, 
        total_docs=val_total_docs, 
        k1=k1, k2=k2, b=b
    )
    
    total_ap = 0.0
    total_ndcg = 0.0
    num_queries = len(val_rankings)
    
    for qid, ideal_qrels in val_rankings.items():
        qid_str = str(qid) # make sure there's no type mismatch
        
        query_scores = bm25_results.get(qid_str, {})
        predicted_pids = sorted(query_scores.keys(), key=lambda x: query_scores[x], reverse=True)
        safe_ideal_qrels = {str(p_id): rel for p_id, rel in ideal_qrels.items()}
        
        total_ap += calculate_average_precision(predicted_pids, safe_ideal_qrels, k=25)
        total_ndcg += calculate_ndcg(predicted_pids, safe_ideal_qrels, k=25)
        
    mean_ap = total_ap / num_queries 
    mean_ndcg = total_ndcg / num_queries 
    
    print(f"MAP@25:  {mean_ap:.4f}")
    print(f"NDCG@25: {mean_ndcg:.4f}")
    print("="*40 + "\n")
    
    return mean_ap, mean_ndcg

def evaluate_logistic_regression(validation_data, weights, bias, model, stop_words):
    print("\nEvaluating logistic regression model...")
    
    # 1. predictions dict { qid: { pid: probability_score } }
    lr_scores = defaultdict(dict)
    
    for _, row in validation_data.iterrows():
        qid = str(row['qid'])
        pid = str(row['pid'])
        
        # Extract features
        query_vec = get_average_embedding(row['queries'], model, stop_words)
        passage_vec = get_average_embedding(row['passage'], model, stop_words)
        feature_vector = np.concatenate([query_vec, passage_vec])
        
        # Predict the probability of relevance
        prob = predict_proba(feature_vector, weights, bias)
        
        # Store the score
        lr_scores[qid][pid] = prob

    # 3. Get ideal rankings from validation data
    val_rankings = get_train_rankings(validation_data)
    
    total_ap = 0.0
    total_ndcg = 0.0
    num_queries = len(val_rankings)
    
    # 4. Calculate MAP and NDCG
    for qid, ideal_qrels in val_rankings.items():
        qid_str = str(qid)
        
        # Get the scores for this query
        query_scores = lr_scores.get(qid_str, {})
        
        # Sort PIDs by predicted probability descending (highest probability first)
        predicted_pids = sorted(query_scores.keys(), key=lambda x: query_scores[x], reverse=True)
        
        # Safe QRELs mapping
        safe_ideal_qrels = {str(p_id): rel for p_id, rel in ideal_qrels.items()}
        
        total_ap += calculate_average_precision(predicted_pids, safe_ideal_qrels, k=25)
        total_ndcg += calculate_ndcg(predicted_pids, safe_ideal_qrels, k=25)
        
    mean_ap = total_ap / num_queries
    mean_ndcg = total_ndcg / num_queries
    
    print(f"Logistic Regression MAP@25:  {mean_ap:.4f}")
    print(f"Logistic Regression NDCG@25: {mean_ndcg:.4f}")
    print("="*40 + "\n")
    
    return mean_ap, mean_ndcg

def get_train_rankings(train_data):
    
    training_rankings = {}  #  qid: { pid: [relevance] } 
    
    for qid, group in train_data.groupby("qid"):
        # sort passages by relevancy score descending
        sorted_group = group.sort_values("relevancy", ascending=False)
        
        training_rankings[qid] = {
            pid: [rel] for pid, rel in zip(sorted_group["pid"], sorted_group["relevancy"])
        }
    
    return training_rankings

def build_index_from_dataframe(df, stop_words):
    # Build inverted index for train_data and validation_data (different structure to other database files)
    print("Building Inverted Index from training data...")
    inverted_index = defaultdict(dict)
    
    # 1. Extract unique passages to avoid indexing the same document twice
    unique_docs = df[['pid', 'passage']].drop_duplicates()
    
    # Optional: Keep track of document lengths here so we don't have to do it later!
    doc_lengths = {}
    
    for _, row in unique_docs.iterrows():
        pid = str(row['pid'])
        
        # Safely convert to string and lower case in case of missing data (NaN)
        passage_text = str(row['passage']).lower()
        passage_text = re.findall(r'[a-z0-9]+', passage_text)
        
        # Count term frequencies for this document
        term_frequencies = defaultdict(int)
        for word in passage_text:
            if word not in stop_words:
                term_frequencies[word] += 1
                
        # Update the Inverted Index
        for word, tf in term_frequencies.items():
            inverted_index[word][pid] = tf
            
        # Store the document length (excluding stop words, just like the index)
        doc_lengths[pid] = sum(term_frequencies.values())
            
    print(f"Index built! Vocabulary size: {len(inverted_index)}")
    print(f"Total unique documents indexed: {len(doc_lengths)}")
    
    return dict(inverted_index), doc_lengths


def get_average_embedding(text, model, stop_words):
    """
    Computes the average word embedding for a given text (query or passage).
    Returns a zero-vector if the text is empty or no words are in the vocabulary.
    """
    # 1. Safely handle empty text or NaN values from pandas
    if pd.isna(text) or not isinstance(text, str):
        return np.zeros(model.vector_size)
        
    # 2. Tokenize and lowercase
    words = text.lower()
    words = re.findall(r'[a-z0-9]+', words)
    
    # 3. Filter out stop words (highly recommended to reduce noise)
    words = [word for word in words if word not in stop_words]
    
    # 4. Extract vectors for words that exist in the model
    vectors = [model[word] for word in words if word in model]
    
    # 5. Fallback: If no words matched (or passage was only stop words)
    # We must return a vector of 0s so the Logistic Regression model doesn't crash
    if not vectors:
        return np.zeros(model.vector_size)
        
    # 6. Calculate the column-wise mean (averages all vectors into one)
    return np.mean(vectors, axis=0)

def train_model(train_data, stop_words):
    model = api.load(MODEL_NAME)

    # --- 1. BALANCED SAMPLING ---
    print("Balancing dataset...")
    
    # Separate the relevant and irrelevant rows
    relevant_df = train_data[train_data['relevancy'] > 0]
    irrelevant_df = train_data[train_data['relevancy'] == 0]
    
    num_relevant = len(relevant_df)
    
    # Sample irrelevant data to be 3x the size of relevant data (1:3 ratio)
    # random_state keeps a set seed so this is reproduceable
    sampled_irrelevant_df = irrelevant_df.sample(n=num_relevant * 3, random_state=42)
    
    # Combine them and shuffle the dataframe (frac=1 means 100% of the data)
    balanced_data = pd.concat([relevant_df, sampled_irrelevant_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Original data size: {len(train_data)}")
    print(f"Balanced data size: {len(balanced_data)} ({num_relevant} relevant, {len(sampled_irrelevant_df)} irrelevant)")

    X_train = []
    y_train = []
    
    print("Extracting feature vectors...")
    for _, row in balanced_data.iterrows():
        query_vec = get_average_embedding(row['queries'], model, stop_words)
        passage_vec = get_average_embedding(row['passage'], model, stop_words)
        
        # concatenate (we get 200 features)
        feature_vector = np.concatenate([query_vec, passage_vec])
        
        label = 1 if row['relevancy'] == 1.0 else 0
        
        X_train.append(feature_vector)
        y_train.append(label)
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"Training data prepared! X shape: {X_train.shape}, y shape: {y_train.shape}")
    
    weights, bias = train_logistic_regression(
        X_train, y_train, 
        learning_rate=1.0, 
        num_iterations=1000   
    )
    
    return weights, bias, model, stop_words

def sigmoid(z):
    #Squashes any number to a probability between 0 and 1.
    # np.clip prevents math overflow errors if z gets too large/small
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred):
    # calculates Binary Cross-Entropy Loss
    epsilon = 1e-9 # Prevents log(0) errors
    y1 = y_true * np.log(y_pred + epsilon)
    y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
    return -np.mean(y1 + y2)

def train_logistic_regression(X, y, learning_rate=0.1, num_iterations=1000):
    #Trains the model using Gradient Descent on the entire dataset
    num_samples, num_features = X.shape
    
    weights = np.zeros(num_features)
    bias = 0.0
    
    print(f"\nStarting Gradient Descent | Iterations: {num_iterations} | Learning Rate= {learning_rate}")
    
    for i in range(num_iterations):
        # forward pass
        linear_model = np.dot(X, weights) + bias
        y_predicted = sigmoid(linear_model)
        
        # backward pass
        dz = y_predicted - y
        dw = (1 / num_samples) * np.dot(X.T, dz)
        db = (1 / num_samples) * np.sum(dz)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        # tsrack loss every 100 iterations and final iteration
        if i % 100 == 0 or i == num_iterations - 1:
            loss = compute_loss(y, y_predicted)
            print(f"Iteration {i:04d} | Loss: {loss:.4f}")
            
    print("Training complete!")
    return weights, bias

def predict_proba(X, weights, bias):
    #Returns the probability that the passage is relevant [0-1].
    linear_model = np.dot(X, weights) + bias
    return sigmoid(linear_model)

if __name__ == "__main__":
    candidate_passages = pd.read_csv(task2.COLLECTION, sep='\t', header=None)
    candidate_passages.columns = ["qid", "pid", "query", "passage"]
    train_data = pd.read_csv("train-data.tsv", sep='\t')
    train_data.columns = ["qid", "pid", "queries", "passage", "relevancy"]
    test_queries = pd.read_csv(TEST_QUERIES, sep='\t', header=None)
    test_queries.columns = ["qid","text"]
    stop_words = set(nltk.corpus.stopwords.words('english'))
    validation_data = pd.read_csv("validation-data.tsv", sep='\t')
    validation_data.columns = ["qid", "pid", "queries", "passage", "relevancy"]
    
    #optimise bm25 params
    #inverted_index, doc_lengths = build_index_from_dataframe(train_data, stop_words)
    #total_docs = len(doc_lengths)
    #avg_dl = sum(doc_lengths.values()) / total_docs if total_docs > 0 else 1.0
    #best_params = optimise_bm25(candidate_passages, test_queries, train_data, inverted_index, doc_lengths, total_docs, avg_dl)

    #evaluate bm25 with optimised params
    #evaluate_bm25(validation_data, inverted_index, doc_lengths, best_params, stop_words)

    #train linear regression model
    weights, bias, ft_model, stop_words=train_model(train_data, stop_words)
    print("CHECKING OUTPUTS")
    print(f"Weights shape: {weights.shape}")
    print(f"Bias value: {bias:.4f}")

    evaluate_logistic_regression(validation_data, weights, bias, ft_model, stop_words)