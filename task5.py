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
    """Calculates Average Precision at cutoff k."""
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

    k1_range = [1.2, 1.5, 1.7]
    k2 = 100
    b_range = [0.65, 0.75, 0.85]
    
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

def evaluate_bm25(validation_data, best_params, stop_words):

    val_inverted_index, val_doc_lengths = build_index_from_dataframe(validation_data, stop_words)
    
    val_total_docs = len(val_doc_lengths)
    val_avg_dl = sum(val_doc_lengths.values()) / val_total_docs if val_total_docs > 0 else 1.0
    
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
        inverted_index=val_inverted_index,           
        candidate_passages=val_candidate_passages, 
        queries_df=val_queries_df, 
        doc_lengths=val_doc_lengths, 
        avg_doc_length=val_avg_dl, 
        total_docs=val_total_docs, 
        k1=k1, k2=k2, b=b
    )
    
    total_ap = 0.0
    total_ndcg = 0.0
    num_queries = len(val_rankings)
    
    for qid, ideal_qrels in val_rankings.items():
        qid_str = str(raw_qid) # make sure there's no type mismatch
        
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
    # embed passages to word2vec, for each query rank top x passages, 
    model = api.load(MODEL_NAME)

    X_train = []
    y_train = []
    for _, row in train_data.iterrows():
        # 1. Get the average embedding for the query
        query_vec = get_average_embedding(row['queries'], model, stop_words)
        
        # 2. Get the average embedding for the passage
        passage_vec = get_average_embedding(row['passage'], model, stop_words)
        
        # 3. Create the feature vector (X)
        # Concatenating them creates a 600-dimensional vector [query_features, passage_features]
        feature_vector = np.concatenate([query_vec, passage_vec])
        
        # 4. Get the label (y)
        # The specification mentions a "binary classification task". 
        # If your relevancy scores are already 0 and 1, we just take the score.
        # If they are graded (e.g., 0 to 3), we need to binarize them (e.g., > 0 is True)
        # Assuming they are binary or graded where > 0 is relevant:
        label = 1 if row['relevancy'] > 0 else 0
        
        X_train.append(feature_vector)
        y_train.append(label)
        
    # Convert lists to numpy arrays for machine learning
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    #max_samples = 100000 
    #if len(X_train) > max_samples:
    #    print(f"Downsampling from {len(X_train)} to {max_samples} for efficiency...")
    #    indices = np.random.choice(len(X_train), max_samples, replace=False)
    #    X_train = X_train[indices]
    #    y_train = y_train[indices]
    
    print(f"Training data prepared! X shape: {X_train.shape}, y shape: {y_train.shape}")
    
    # Train the model using our functional implementation
    chosen_learning_rate = 0.1
    iterations = 1000
    
    weights, bias = train_logistic_regression(
        X_train, y_train, 
        learning_rate=chosen_learning_rate, 
        num_iterations=iterations
    )
    
    # Return the learned parameters alongside the FastText model and stop words
    # We will need all of these to evaluate the validation set later!
    return weights, bias, model, stop_words

def sigmoid(z):
    """Squashes any number to a probability between 0 and 1."""
    # np.clip prevents math overflow errors if z gets too large/small
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred):
    """Calculates Binary Cross-Entropy Loss."""
    epsilon = 1e-9 # Prevents log(0) errors
    y1 = y_true * np.log(y_pred + epsilon)
    y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
    return -np.mean(y1 + y2)

def train_logistic_regression(X, y, learning_rate=0.1, num_iterations=1000):
    """Trains the model using Gradient Descent and returns the optimal weights and bias."""
    num_samples, num_features = X.shape
    
    # Initialize weights and bias to zeros
    weights = np.zeros(num_features)
    bias = 0.0
    
    print(f"\nStarting training with Learning Rate: {learning_rate}")
    
    # Gradient Descent Loop
    for i in range(num_iterations):
        # --- Forward Pass ---
        linear_model = np.dot(X, weights) + bias
        y_predicted = sigmoid(linear_model)
        
        # --- Backward Pass (Gradients) ---
        dz = y_predicted - y
        dw = (1 / num_samples) * np.dot(X.T, dz)
        db = (1 / num_samples) * np.sum(dz)
        
        # --- Update Weights ---
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        # Track and print loss every 100 iterations
        if i % 100 == 0:
            loss = compute_loss(y, y_predicted)
            print(f"Iteration {i:04d} | Loss: {loss:.4f}")
            
    print("Training complete!")
    return weights, bias

def predict_proba(X, weights, bias):
    """Returns the probability that the passage is relevant (0 to 1)."""
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
    #inverted_index, doc_lengths = build_index_from_dataframe(train_data)

    #best_params = optimise_bm25(candidate_passages, test_queries, train_data, inverted_index, doc_lengths, total_docs, avg_dl)

    validation_data = pd.read_csv("validation-data.tsv", sep='\t')
    validation_data.columns = ["qid", "pid", "queries", "passage", "relevancy"]
    #best_params = {"k1": 1.2, "k2": 100, "b": 0.75}
    #evaluate_bm25(validation_data, best_params, stop_words)

    weights, bias, ft_model, stop_word=train_model(train_data, stop_words)
    print("\n--- CHECKING OUTPUTS ---")
    print(f"Weights shape: {weights.shape} (Should be 200)")
    print(f"Bias value:    {bias:.4f}")