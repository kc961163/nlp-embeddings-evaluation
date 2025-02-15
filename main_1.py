# Import necessary libraries
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import math
import wordninja

from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# For stopwords, we will use NLTK.
# Uncomment the two lines below if you haven't already downloaded stopwords.
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

# Set of English stopwords
stop_words = set(stopwords.words('english'))

# Global cache for word segmentation
word_segmentation_cache = {}

# Global debug flag to control debug prints
DEBUG = True

# -------------------------------------------------------------------
# Function: load_data
# -------------------------------------------------------------------
def load_data(path):
    """
    Load the CSV data from the given path.
    
    Parameters:
        path (str): The file path to the CSV data.
    
    Returns:
        df (DataFrame): Pandas DataFrame containing the loaded data.
    """
    df = pd.read_csv(path)
    if DEBUG:
        print("DEBUG: Loaded data with shape:", df.shape)
        print("DEBUG: First few rows:")
        print(df.head())
    return df

# -------------------------------------------------------------------
# Function: preprocess_authors
# -------------------------------------------------------------------
def preprocess_authors(df):
    """
    Preprocess the 'author' column by converting to lower case, stripping whitespace,
    and—if a comma is present—keeping only the part before the comma.
    Non-string values are converted to an empty string.
    
    Parameters:
        df (DataFrame): The DataFrame containing the 'author' column.
    
    Returns:
        df (DataFrame): The DataFrame with a cleaned 'author' column.
    """
    def clean_author(x):
        if isinstance(x, str):
            # Lowercase and strip whitespace
            x = x.strip().lower()
            # If there is a comma, assume extra info follows; keep only the first part.
            if ',' in x:
                x = x.split(',')[0].strip()
            return x
        else:
            return ""
    
    df['author'] = df['author'].apply(clean_author)
    if DEBUG:
        print("\nDEBUG: Completed author preprocessing.")
        print("DEBUG: Unique authors after preprocessing (sample):", df['author'].unique()[:10])
    return df

# -------------------------------------------------------------------
# Function: preprocess_text
# -------------------------------------------------------------------
def preprocess_text(text):
    """
    Preprocess a single text string by:
      - Converting to lowercase.
      - Removing punctuation (replacing it with a space).
      - Splitting the text into tokens.
      - For tokens longer than a threshold (5 characters), attempt to split them using wordninja.
      - Removing stopwords.
    
    Parameters:
        text (str): The text to preprocess.
    
    Returns:
        cleaned_text (str): The cleaned text as a string.
        filtered_words (list): A list of the final tokens.
    """
    # If not a string (e.g., NaN), return empty values.
    if not isinstance(text, str):
        if DEBUG:
            print("DEBUG: Non-string text encountered. Value:", text)
        return "", []
    
    # Step 1: Lowercase the text.
    text = text.lower()
    
    # Step 2: Remove punctuation by replacing punctuation with a space.
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    # Step 3: Split the text on whitespace.
    words = text.split()
    
    # Step 4: For tokens longer than the threshold, attempt segmentation with wordninja (using caching).
    threshold = 5  # Adjust this value as needed.
    segmented_words = []
    for word in words:
        if len(word) > threshold:
            # Check if we have already segmented this word.
            if word in word_segmentation_cache:
                splits = word_segmentation_cache[word]
            else:
                splits = wordninja.split(word)
                word_segmentation_cache[word] = splits
            # If segmentation returns more than one token, use them.
            if len(splits) > 1:
                segmented_words.extend(splits)
            else:
                segmented_words.append(word)
        else:
            segmented_words.append(word)
    
    # Step 5: Remove stopwords. 
    filtered_words = [word for word in segmented_words if word not in stop_words]
    
    # Step 6: Rejoin tokens into a cleaned text string.
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text, filtered_words

# -------------------------------------------------------------------
# Function: preprocess_dataframe
# -------------------------------------------------------------------
def preprocess_dataframe(df):
    """
    Apply text preprocessing to the 'quote' column of the DataFrame.
    
    Parameters:
        df (DataFrame): DataFrame containing at least the 'quote' column.
    
    Returns:
        df (DataFrame): DataFrame with additional 'cleaned_quote' and 'tokens' columns.
    """
    _, df['tokens'] = zip(*df['quote'].apply(preprocess_text))
    if DEBUG:
        print("\nDEBUG: Completed text preprocessing for all quotes.")
    return df

# -------------------------------------------------------------------
# Function: group_quotes_by_author
# -------------------------------------------------------------------
def group_quotes_by_author(df):
    """
    Group quotes by author. Combine all quotes for an author by merging their token lists.
    
    Parameters:
        df (DataFrame): DataFrame with 'author' and 'tokens' columns.
    
    Returns:
        grouped (DataFrame): DataFrame grouped by 'author' with a combined 'tokens' list.
    """
    grouped = df.groupby('author').agg({
        'tokens': lambda token_lists: [token for tokens in token_lists for token in tokens]
    }).reset_index()
    
    if DEBUG:
        print(f"\nDEBUG: Grouped documents by author (tokens only). Total authors: {len(grouped)}")
        print("DEBUG: Sample authors:")
        print(grouped[['author']].head())
    return grouped

# -------------------------------------------------------------------
# Function: select_top_words
# -------------------------------------------------------------------
def select_top_words(grouped, selected_authors=None, min_length=4, top_n=2):
    """
    For a set of selected authors, identify the top N most frequent words 
    (with a minimum specified length) from their combined tokens.
    
    Parameters:
        grouped (DataFrame): DataFrame with grouped quotes by author.
        selected_authors (list): List of authors to analyze. If None, take the first 3.
        min_length (int): Minimum length of words to consider.
        top_n (int): Number of top words to return per author.
    
    Returns:
        top_words (dict): Dictionary mapping each author to a list of (word, frequency) tuples.
    """
    if selected_authors is None:
        # Since empty authors have been dropped, simply take the first three.
        selected_authors = grouped['author'].head(3).tolist()
    
    if DEBUG:
        print("\nDEBUG: Selected authors for frequency analysis:", selected_authors)
    
    top_words = {}
    for author in selected_authors:
        tokens = grouped.loc[grouped['author'] == author, 'tokens'].values[0]
        tokens_filtered = [word for word in tokens if len(word) >= min_length]
        word_counts = Counter(tokens_filtered)
        most_common = word_counts.most_common(top_n)
        top_words[author] = most_common
        print(f"\nTop {top_n} words for author '{author}': {most_common}")
    return top_words, selected_authors

# -------------------------------------------------------------------
# Function: compute_tf_idf
# -------------------------------------------------------------------
def compute_tf_idf(grouped, min_length=4):
    """
    Compute the TF-IDF values for each author's document.
    
    Parameters:
        grouped (DataFrame): Grouped DataFrame by author.
        min_length (int): Minimum word length to consider.
    
    Returns:
        author_tfidf (dict): Dictionary mapping author names to their TF-IDF dictionary.
    """
    author_tf = {}
    doc_freq = {}
    N = len(grouped)
    
    if DEBUG:
        print(f"\nDEBUG: Total number of author documents: {N}")
    
    # Calculate Term Frequency (TF) and Document Frequency (DF)
    # Loop through each author and their combined tokens
    for idx, row in grouped.iterrows():
        author = row['author']
        tokens = row['tokens']
        # Filter out words that are shorter than the minimum length
        tokens_filtered = [word for word in tokens if len(word) >= min_length]
        # Count occurrences of each word
        tf_counter = Counter(tokens_filtered)
        # Sum all word counts to get the total number of words
        total_words = sum(tf_counter.values())
        # Term Frequency (TF) = (count of each word) / (total words for the author)
        author_tf[author] = {word: count / total_words for word, count in tf_counter.items()}
        # Document Frequency (DF) increases by 1 if a word appears in an author's document
        for word in tf_counter.keys():
            doc_freq[word] = doc_freq.get(word, 0) + 1

    # Compute Inverse Document Frequency (IDF)
    idf = {word: math.log((N + 1) / (freq + 1)) + 1 for word, freq in doc_freq.items()}
    
    if DEBUG:
        sample_idf = dict(list(idf.items())[:5])
        print("\nDEBUG: Sample IDF values:")
        for word, value in sample_idf.items():
            print(f"  Word: '{word}' -> IDF: {value:.6f}")
    
    # Compute TF-IDF for each author
    def compute_tfidf_for_doc(tf_dict):
        return {word: tf_value * idf.get(word, 0) for word, tf_value in tf_dict.items()}
    
    author_tfidf = {author: compute_tfidf_for_doc(tf_dict) for author, tf_dict in author_tf.items()}
    return author_tfidf

# -------------------------------------------------------------------
# Function: compare_tfidf_with_sklearn
# -------------------------------------------------------------------
def compare_tfidf_with_sklearn(grouped):
    """
    Compute TF-IDF using scikit-learn's TfidfVectorizer on the combined author documents,
    then compare the top words from scikit-learn with those from the custom TF-IDF.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Convert tokens (lists) to strings by joining them with a space
    documents = [' '.join(tokens) for tokens in grouped['tokens'].tolist()]
    authors = grouped['author'].tolist()

    # Initialize TfidfVectorizer (using the same stop words)
    vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r"(?u)\b\w{4,}\b")
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert the TF-IDF matrix to a DataFrame.
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), index=authors, columns=feature_names)
    print("\nTF-IDF matrix from scikit-learn (first 5 rows):")
    print(df_tfidf.head())

    # Get custom TF-IDF using our method with min_length=4
    custom_tfidf = compute_tf_idf(grouped, min_length=4)

    # For comparison, print the top 5 words (by custom TF-IDF) for the first few authors.
    for author in authors[:3]:
        print(f"\nCustom TF-IDF top words for author '{author}':")
        sorted_words = sorted(custom_tfidf[author].items(), key=lambda x: x[1], reverse=True)
        print(sorted_words[:5])
        print(f"\nscikit-learn TF-IDF top words for author '{author}':")
        # For scikit-learn, get the row corresponding to the author and sort
        author_series = df_tfidf.loc[author]
        top_words_sklearn = author_series.sort_values(ascending=False).head(5)
        print(top_words_sklearn)

# -------------------------------------------------------------------
# Function: train_word2vec_model
# -------------------------------------------------------------------
def train_word2vec_model(corpus, window, vector_size, negative, seed=42):
    """
    Train a Word2Vec model on the given corpus with the specified parameters.
    
    Parameters:
        corpus (list of list of str): The training data, a list of tokenized sentences.
        window (int): The context window size.
        vector_size (int): The number of dimensions of the word vectors.
        negative (int): The number of negative samples to use.
        seed (int): Random seed for reproducibility.
    
    Returns:
        model (Word2Vec): The trained Word2Vec model.
    """
    model = Word2Vec(
        sentences=corpus,
        sg=1,                # Use skip-gram
        window=window,
        vector_size=vector_size,  # For gensim version >= 4.0 use vector_size; for older versions use size
        negative=negative,
        min_count=5,         # Ignore words that appear less than 5 times
        workers=4,
        seed=seed
    )
    return model

# -------------------------------------------------------------------
# Function: tsne_visualize_embeddings
# -------------------------------------------------------------------
def tsne_visualize_embeddings(model, words, title="TSNE Visualization", save_path=None):
    """
    Create a TSNE visualization for the embeddings of the given words.
    
    Parameters:
        model (Word2Vec): A trained Word2Vec model.
        words (list of str): List of key words to visualize.
        title (str): Title for the plot.
        save_path (str): Optional file path to save the plot. If provided, the plot is saved.
    
    Returns:
        None: The plot is saved (if save_path is provided) or displayed.
    """
    word_vectors = []
    valid_words = []
    for word in words:
        if word in model.wv.key_to_index:
            word_vectors.append(model.wv[word])
            valid_words.append(word)
        else:
            print(f"WARNING: '{word}' not in vocabulary.")
    
    if not word_vectors:
        print("No valid words to visualize.")
        return
    
    word_vectors = np.array(word_vectors)
    tsne = TSNE(n_components=2, random_state=42, perplexity=3)
    reduced_vectors = tsne.fit_transform(word_vectors)
    
    plt.figure(figsize=(8, 6))
    for i, word in enumerate(valid_words):
        x, y = reduced_vectors[i, 0], reduced_vectors[i, 1]
        plt.scatter(x, y, marker='o', color='red')
        plt.annotate(word, (x, y), textcoords="offset points", xytext=(5,5), ha='right')
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# -------------------------------------------------------------------
# Function: tsne_visualize_similar_groups
# -------------------------------------------------------------------
def tsne_visualize_similar_groups(model, key_words, topn=3, combined=False, save_path=None):
    """
    For each key word, retrieve its top 'topn' similar words (including itself) and visualize them with TSNE.
    
    Parameters:
        model: A trained Word2Vec model.
        key_words (list of str): List of key words to analyze.
        topn (int): Number of similar words to retrieve.
        combined (bool): If True, plot all groups in one combined plot; if False, create separate subplots.
        save_path (str): Optional file path to save the plot. If provided, the plot is saved.
    
    Returns:
        None: The plot is saved (if save_path is provided) or displayed.
    """
    groups = {}
    for word in key_words:
        if word in model.wv.key_to_index:
            similar = model.wv.most_similar(positive=[word], topn=topn)
            group_words = [word] + [w for w, score in similar]
            groups[word] = group_words
        else:
            print(f"WARNING: '{word}' not in vocabulary.")
    
    if combined:
        all_words = []
        group_labels = []
        for key, words in groups.items():
            all_words.extend(words)
            group_labels.extend([key] * len(words))
        vectors = np.array([model.wv[w] for w in all_words])
        tsne = TSNE(n_components=2, random_state=42, perplexity=3)
        reduced = tsne.fit_transform(vectors)
        plt.figure(figsize=(10, 8))
        unique_labels = list(set(group_labels))
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        for label, color in zip(unique_labels, colors):
            indices = [i for i, l in enumerate(group_labels) if l == label]
            plt.scatter(reduced[indices, 0], reduced[indices, 1], color=color, label=label)
            for i in indices:
                plt.annotate(all_words[i], (reduced[i, 0], reduced[i, 1]), textcoords="offset points", xytext=(5,5))
        plt.title("Combined TSNE Visualization of Key Words and Their Top Similar Words")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Combined plot saved to {save_path}")
        else:
            plt.show()
    else:
        n = len(groups)
        fig, axs = plt.subplots(1, n, figsize=(5*n, 5))
        if n == 1:
            axs = [axs]
        for ax, (key, words) in zip(axs, groups.items()):
            vectors = np.array([model.wv[w] for w in words])
            tsne = TSNE(n_components=2, random_state=42, perplexity=3)
            reduced = tsne.fit_transform(vectors)
            ax.scatter(reduced[:, 0], reduced[:, 1], color='red')
            for i, w in enumerate(words):
                ax.annotate(w, (reduced[i, 0], reduced[i, 1]), textcoords="offset points", xytext=(5,5))
            ax.set_title(f"{key} (plus top {topn} similar)")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Separate subplots saved to {save_path}")
        else:
            plt.show()

# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main():
    # Task 1.2.1: Load, preprocess, and group the data.
    data_path = './archive/quotes.csv'  # Update this path as needed.
    df = load_data(data_path)
    df = preprocess_authors(df)
    df = df[df['author'].str.strip() != '']  # Drop rows with empty author names.
    df = preprocess_dataframe(df)
    # Save filtered data (only authors and tokens) to a new CSV file
    df[['author', 'tokens']].to_csv('./filtered_quotes.csv', index=False)
    grouped = group_quotes_by_author(df)
    grouped[['author', 'tokens']].to_csv('./grouped_quotes.csv', index=False)
    
    # Task 1.2.2: Select 3 random authors and identify their top 2 frequent words,
    # then compute TF-IDF scores for these words.
    selected_authors = ['albert einstein', 'abraham lincoln', 'mark twain']
    top_words, selected_authors = select_top_words(grouped, selected_authors)
    author_tfidf = compute_tf_idf(grouped)
    
    print("\nTF-IDF scores for the top words in the selected authors' documents:")
    for author in selected_authors:
        print(f"\nAuthor: {author}")
        for word, freq in top_words[author]:
            tfidf_score = author_tfidf[author].get(word, 0)
            print(f"  Word: '{word}' | Frequency: {freq} | TF-IDF: {tfidf_score:.6f}")
    
    # # Compare our custom TF-IDF with scikit-learn's TfidfVectorizer.
    # compare_tfidf_with_sklearn(grouped)
    
    # # --- Task 1.2.3: Train Word2Vec Skip-gram Models ---
    
    # Prepare the corpus from individual quotes (list of token lists)
    corpus = df['tokens'].tolist()
    
    # Train Model A: [Context window size: 3, Vector dimensions: 50, Negative samples: 3]
    model_A = train_word2vec_model(corpus, window=3, vector_size=50, negative=3)
    
    # Train Model B: [Context window size: 7, Vector dimensions: 150, Negative samples: 10]
    model_B = train_word2vec_model(corpus, window=7, vector_size=150, negative=10)
    
    # Compare the models:
    print("\nComparing Word2Vec Models on 'imagination':")
    similar_A = model_A.wv.most_similar(positive=["imagination"])
    similar_B = model_B.wv.most_similar(positive=["imagination"])
    print("Configuration A (window=3, size=50, negative=3):", similar_A)
    print("Configuration B (window=7, size=150, negative=10):", similar_B)
    
    print("\nComparing Word2Vec Models on similarity between 'fear' and 'knowledge':")
    sim_A = model_A.wv.similarity("fear", "knowledge")
    sim_B = model_B.wv.similarity("fear", "knowledge")
    print("Configuration A similarity:", sim_A)
    print("Configuration B similarity:", sim_B)

    # Task 1.2.4: TSNE Visualization of Key Word Embeddings.
    key_words = ["imagination", "fear", "understanding", "discovery", "science"]
    
    print("\nTSNE Visualization for Configuration A:")
    tsne_visualize_embeddings(model_A, key_words, title="TSNE Visualization (Config A)", save_path="tsne_configA.png")
    
    print("\nTSNE Visualization for Configuration B:")
    tsne_visualize_embeddings(model_B, key_words, title="TSNE Visualization (Config B)", save_path="tsne_configB.png")
    
    # Additional: TSNE visualization for similar word groups.
    print("\nTSNE Visualization for Similar Word Groups (Combined) - Model A:")
    tsne_visualize_similar_groups(model_A, key_words, topn=3, combined=True, save_path="tsne_similar_combined_A.png")
    
    print("\nTSNE Visualization for Similar Word Groups (Combined) - Model B:")
    tsne_visualize_similar_groups(model_B, key_words, topn=3, combined=True, save_path="tsne_similar_combined_B.png")

    print("\nTSNE Visualization for Similar Word Groups (Separate Subplots) - Model A:")
    tsne_visualize_similar_groups(model_A, key_words, topn=3, combined=False, save_path="tsne_similar_separate_A.png")
    
    print("\nTSNE Visualization for Similar Word Groups (Separate Subplots) - Model B:")
    tsne_visualize_similar_groups(model_B, key_words, topn=3, combined=False, save_path="tsne_similar_separate_B.png")

# -------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
