import gensim.downloader as api
import numpy as np
import math
import re
import string
from collections import Counter
import matplotlib.pyplot as plt  # (for TSNE visualization if needed)
import sys

# Global debug flag
DEBUG = True

# -------------------------------------------------------------------
# Function: parse_analogy_dataset
# -------------------------------------------------------------------
def parse_analogy_dataset(file_path, desired_groups):
    """
    Parse the analogy dataset file (e.g., word-test.v1.txt) and return a dictionary
    mapping each desired group to a list of analogy questions.
    
    Each analogy question is a tuple of four words (a, b, c, d).

    Parameters:
        file_path (str): Path to the analogy dataset file.
        desired_groups (set): Set of group names (e.g., {"capital-common-countries", ...}).

    Returns:
        analogy_data (dict): Dictionary with keys as group names and values as lists of tuples.
    """
    analogy_data = {}
    current_group = None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                group_name = line[1:].strip()
                if group_name in desired_groups:
                    current_group = group_name
                    analogy_data[current_group] = []
                else:
                    current_group = None
                    print(f"Skipping group '{group_name}'...")
            else:
                if current_group is not None:
                    parts = line.split()
                    if len(parts) == 4:
                        analogy_data[current_group].append(tuple(parts))
    if DEBUG:
        print("\nDEBUG: Parsed analogy dataset. Groups and question counts:")
        for group, questions in analogy_data.items():
            print(f"  {group}: {len(questions)} questions")
    return analogy_data

# -------------------------------------------------------------------
# Non-Optimized Analogy Test Function
# -------------------------------------------------------------------
def run_analogy_test(model, analogy_data):
    """
    Run the analogy prediction test on a pretrained model using gensim's most_similar().
    For each analogy question (a, b, c, d), predict d and count correct answers.

    Returns:
        group_results (dict): Group-wise accuracy.
        overall_accuracy (float): Overall accuracy.
    """
    correct = 0
    total = 0
    group_results = {}
    for group, questions in analogy_data.items():
        group_correct = 0
        for (a, b, c, d) in questions:
            if not all(word in model.key_to_index for word in [a, b, c, d]):
                continue
            try:
                prediction = model.most_similar(positive=[b, c], negative=[a], topn=1)
            except Exception as e:
                if DEBUG:
                    print(f"DEBUG: Exception for analogy ({a}, {b}, {c}, {d}): {e}")
                continue
            predicted_word, score = prediction[0]
            if predicted_word == d:
                group_correct += 1
            total += 1
        accuracy = (group_correct / len(questions)) * 100 if questions else 0
        group_results[group] = accuracy
        if DEBUG:
            print(f"DEBUG: Group '{group}' accuracy (non-opt): {accuracy:.2f}% ({group_correct}/{len(questions)})")
        correct += group_correct
    overall_accuracy = (correct / total) * 100 if total else 0
    return group_results, overall_accuracy

# -------------------------------------------------------------------
# Optimized Analogy Test Functions
# -------------------------------------------------------------------
# def vectorized_analogy(model, a, b, c, topn=1):
#     """
#     Compute analogy prediction in a vectorized manner.
#     Given words a, b, c, compute target = v_b - v_a + v_c and return the topn words
#     whose vectors have highest cosine similarity to the target.
    
#     Assumes that model.vectors_norm is already computed.
    
#     Returns:
#         List of tuples: [(predicted_word, similarity), ...]
#     """
#     # Check if the required words are in the model.
#     if not all(word in model.key_to_index for word in [a, b, c]):
#         return []
#     va = model[a]
#     vb = model[b]
#     vc = model[c]
#     target = vb - va + vc
#     norm_target = np.linalg.norm(target)
#     if norm_target == 0:
#         return []
#     target = target / norm_target

#     # Use precomputed normalized vectors using get_normed_vectors() for gensim 4.x.
#     normed_vectors = model.get_normed_vectors()
    
#     # Compute cosine similarities via dot product.
#     similarities = np.dot(normed_vectors, target)
    
#     # Exclude the input words from the candidate set.
#     for word in [a, b, c]:
#         if word in model.key_to_index:
#             idx = model.key_to_index[word]
#             similarities[idx] = -np.inf
    
#     best_indices = np.argsort(similarities)[-topn:][::-1]
#     best_words = [(model.index_to_key[i], similarities[i]) for i in best_indices]
#     return best_words

# def run_analogy_test_optimized(model, analogy_data):
#     """
#     Run the analogy test using the vectorized_analogy function.
#     Precompute the normalized vectors once for the model to speed up processing.
    
#     Returns:
#         group_results (dict): Group-wise accuracy.
#         overall_accuracy (float): Overall accuracy.
#     """
#     # Precompute normalized vectors once (if not already computed).
#     if not hasattr(model, 'vectors_norm'):
#         model.vectors_norm = model.get_normed_vectors()
    
#     correct = 0
#     total = 0
#     group_results = {}
#     for group, questions in analogy_data.items():
#         group_correct = 0
#         for (a, b, c, d) in questions:
#             if not all(word in model.key_to_index for word in [a, b, c, d]):
#                 continue
#             prediction = vectorized_analogy(model, a, b, c, topn=1)
#             if prediction:
#                 predicted_word, score = prediction[0]
#                 if predicted_word == d:
#                     group_correct += 1
#                 total += 1
#         # Compute accuracy for the group as (correct / total questions in the group) * 100.
#         accuracy = (group_correct / len(questions)) * 100 if questions else 0
#         group_results[group] = accuracy
#         if DEBUG:
#             print(f"DEBUG: Group '{group}' accuracy (opt): {accuracy:.2f}% ({group_correct}/{len(questions)})")
#         correct += group_correct
#     overall_accuracy = (correct / total) * 100 if total else 0
#     return group_results, overall_accuracy

# -------------------------------------------------------------------
# Function: run_antonym_test
# -------------------------------------------------------------------
def run_antonym_test(model, test_words, topn=10):
    if DEBUG:
        print(f"\nDEBUG: Running run_antonym_test with topn={topn} and {len(test_words)} test words.")
    for word in test_words:
        # if DEBUG:
        #     print(f"DEBUG: Checking word '{word}' in vocabulary.")
        if word in model.key_to_index:
            similar = model.most_similar(word, topn=topn)
            print(f"\nTop {topn} words similar to '{word}':")
            for sim_word, sim_score in similar:
                print(f"  {sim_word}: {sim_score:.4f}")
        else:
            print(f"WARNING: '{word}' not in vocabulary.")
            if DEBUG:
                print(f"DEBUG: Skipped word '{word}' as it is not in the vocabulary.")

# -------------------------------------------------------------------
# Function: Custom Analogy Test Functions
# -------------------------------------------------------------------
def run_custom_analogy_test(model, custom_questions):
    """
    Evaluate a list of custom analogy questions on a given pretrained model.
    
    Each custom question is a tuple of four words: (a, b, c, d)
    representing the analogy a:b :: c:d.
    
    Returns:
        accuracy (float): Percentage of correctly answered questions.
    """
    correct = 0
    total = len(custom_questions)
    for (a, b, c, d) in custom_questions:
        if not all(word in model.key_to_index for word in [a, b, c, d]):
            if DEBUG:
                print(f"DEBUG: Skipping question ({a}, {b}, {c}, {d}) because one or more words are not in the vocabulary.")
            continue
        try:
            prediction = model.most_similar(positive=[b, c], negative=[a], topn=1)
        except Exception as e:
            if DEBUG:
                print(f"DEBUG: Exception for custom analogy ({a}, {b}, {c}, {d}): {e}")
            continue
        predicted_word, score = prediction[0]
        if predicted_word == d:
            correct += 1
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy

# -------------------------------------------------------------------
# Function: load_pretrained_models (using gensim.downloader)
# -------------------------------------------------------------------
def load_pretrained_models():
    """
    Download and load two pretrained models using gensim.downloader.
    Returns:
        model1, model2: Pretrained embedding models.
    """
    print("\nDownloading pretrained model: word2vec-google-news-300")
    model1 = api.load("word2vec-google-news-300")  # This returns a KeyedVectors instance.
    
    print("\nDownloading pretrained model: glove-wiki-gigaword-300")
    model2 = api.load("glove-wiki-gigaword-300")      # This also returns a KeyedVectors instance.
    
    return model1, model2

# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main():
    # --- Task 2.1: Analogy Test with Pretrained Embeddings ---
    
    # Desired analogy groups.
    desired_groups = {
        "capital-common-countries", 
        "currency", 
        "city-in-state", 
        "family", 
        "gram1-adjective-to-adverb", 
        "gram2-opposite", 
        "gram3-comparative", 
        "gram6-nationality-adjective"
    }
    
    # Parse the analogy dataset.
    analogy_file = "./word-test.v1.txt"  # Update path as needed.
    analogy_data = parse_analogy_dataset(analogy_file, desired_groups)
    
    # Load pretrained models.
    model1, model2 = load_pretrained_models()
    
    # Run the non-optimized analogy test on both models.
    # (These lines are commented out; for now, we are using the optimized version.)
    print("\n--- Running Non-Optimized Analogy Test ---")
    print("\nModel 1 (Google News Word2Vec):")
    group_results1, overall_accuracy1 = run_analogy_test(model1, analogy_data)
    print("Group-wise accuracy for Model 1:", group_results1)
    print(f"Overall accuracy for Model 1: {overall_accuracy1:.2f}%")
    
    print("\nModel 2 (GloVe Wiki Gigaword):")
    group_results2, overall_accuracy2 = run_analogy_test(model2, analogy_data)
    print("Group-wise accuracy for Model 2:", group_results2)
    print(f"Overall accuracy for Model 2: {overall_accuracy2:.2f}%")
    
    # Run the optimized analogy test on both models.
    # print("\n--- Running Optimized Analogy Test ---")
    # print("\nModel 1 (Google News Word2Vec):")
    # group_results1_opt, overall_accuracy1_opt = run_analogy_test_optimized(model1, analogy_data)
    # print("Group-wise accuracy (optimized) for Model 1:", group_results1_opt)
    # print(f"Overall accuracy (optimized) for Model 1: {overall_accuracy1_opt:.2f}%")
    
    # print("\nModel 2 (GloVe Wiki Gigaword):")
    # group_results2_opt, overall_accuracy2_opt = run_analogy_test_optimized(model2, analogy_data)
    # print("Group-wise accuracy (optimized) for Model 2:", group_results2_opt)
    # print(f"Overall accuracy (optimized) for Model 2: {overall_accuracy2_opt:.2f}%")

    # --- Task 2.2: Antonym Test with Pretrained Embeddings ---
    
    # Define a list of test words that have clear antonyms.
    # We include multiple words to cover a range of cases.
    test_words = ["increase", "enter", "happy", "hot", "light", "big", "fast"]
    # For instance:
    #   "increase" should be opposed by "decrease"
    #   "enter" should be opposed by "exit"
    #   "happy" vs. "sad"
    #   "hot" vs. "cold"
    #   "light" vs. "dark"
    #   "big" vs. "small"
    #   "fast" vs. "slow"
    
    # Run the antonym test on Model 1.
    print("\n--- Running Antonym Test on Model 1 (Google News Word2Vec) ---")
    run_antonym_test(model1, test_words, topn=10)
    
    # Run the antonym test on Model 2.
    print("\n--- Running Antonym Test on Model 2 (GloVe Wiki Gigaword) ---")
    run_antonym_test(model2, test_words, topn=10)

    # --- Task 2.3: Custom Analogy Tests ---
    # Define two sets of custom analogy questions (each question is a tuple (a, b, c, d)).
    
    # Category A: Domain-Specific Analogies (e.g., Technology/Finance)
    custom_domain_analogies = [
        # Example 1: Technology domain
        ("server", "data", "library", "books"),
        # Example 2: Technology domain
        ("smartphone", "communication", "laptop", "computing"),
        # Example 3: Finance domain
        ("investment", "risk", "savings", "security")
    ]
    
    # Category B: Adversarial/Contrastive Analogies
    custom_adversarial_analogies = [
        ("up", "down", "fast", "slow"),
        ("light", "dark", "high", "low"),
        ("big", "small", "old", "young")
    ]
    
    # Evaluate custom analogies on both models.
    print("\n--- Running Custom Analogy Test: Category A (Domain-Specific) ---")
    acc_domain_model1 = run_custom_analogy_test(model1, custom_domain_analogies)
    acc_domain_model2 = run_custom_analogy_test(model2, custom_domain_analogies)
    print(f"Model 1 accuracy on Domain-Specific Analogies: {acc_domain_model1:.2f}%")
    print(f"Model 2 accuracy on Domain-Specific Analogies: {acc_domain_model2:.2f}%")
    
    print("\n--- Running Custom Analogy Test: Category B (Adversarial/Contrastive) ---")
    acc_adv_model1 = run_custom_analogy_test(model1, custom_adversarial_analogies)
    acc_adv_model2 = run_custom_analogy_test(model2, custom_adversarial_analogies)
    print(f"Model 1 accuracy on Adversarial Analogies: {acc_adv_model1:.2f}%")
    print(f"Model 2 accuracy on Adversarial Analogies: {acc_adv_model2:.2f}%")

if __name__ == "__main__":
    main()