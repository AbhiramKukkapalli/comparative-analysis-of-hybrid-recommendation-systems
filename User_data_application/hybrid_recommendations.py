import streamlit as st
import streamlit.components.v1 as components
import json
import pandas as pd
from datetime import datetime
import random
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from indicnlp.tokenize import indic_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import string
import regex 
import ast
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import numpy as np


# Preprocessing function
def preprocess_data(df):
    df["body_processed"] = df["body"].str.replace('\u200c', '').str.replace('\n', '').str.replace('\t', '').str.replace('\xa0', '')
    df["body_processed"] = df["body_processed"].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))
    return df

# Custom analyzer for CountVectorizer using indic_tokenize
def custom_analyzer(text):
    # Tokenize the text using indic_tokenize
    words = indic_tokenize.trivial_tokenize(text, lang='te')  # Assuming the text is in Telugu
    for w in words:
        yield w


def find_article_index(sno, articles_df):
    indices = articles_df.index[articles_df['SNo'] == int(sno)].tolist()
    return indices[0] if indices else None

def find_user_index(user_id, user_data_df):
    print("Length of DataFrame:", len(user_data_df))
    matches = user_data_df[user_data_df['user_id'] == user_id]
    print(matches.index[0])
    return matches.index[0]-1 if not matches.empty else None


# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    train_path = "./train_telugu_news.csv"
    test_path = "./test_telugu_news.csv"
    telugu_news_df = pd.read_csv(train_path)
    test_news_df = pd.read_csv(test_path)

    # Preprocess steps
    # Removing special characters and unnecessary white spaces
    telugu_news_df["body_processed"] = telugu_news_df["body"].str.replace('\u200c', '').str.replace('\n', '').str.replace('\t', '').str.replace('\xa0', '')
    telugu_news_df["body_processed"] = telugu_news_df["body_processed"].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))

    test_news_df["body_processed"] = test_news_df["body"].str.replace('\u200c', '').str.replace('\n', '').str.replace('\t', '').str.replace('\xa0', '')
    test_news_df["body_processed"] = test_news_df["body_processed"].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))

    # Categorical Encoding of Topics
    topic_dic = {topic: idx for idx, topic in enumerate(telugu_news_df["topic"].unique())}
    inv_topic_dict = {v: k for k, v in topic_dic.items()}
    
    telugu_news_df["topic"] = telugu_news_df["topic"].map(topic_dic)
    test_news_df["topic"] = test_news_df["topic"].map(topic_dic)

    return telugu_news_df, test_news_df, inv_topic_dict

telugu_news_df, test_news_df, inv_topic_dict = load_and_preprocess_data()


# Model training and feature extraction code
# ...
# Feature Extraction - Preparing the training and testing data
categories = [i for i in range(len(telugu_news_df["topic"].unique()))]
text_topic = [' '.join(telugu_news_df[telugu_news_df["topic"] == i]["body_processed"].tolist()) for i in categories]

# Using CountVectorizer for feature extraction
count_vec = CountVectorizer(max_df=0.75, min_df=0.1, lowercase=False, analyzer=custom_analyzer, max_features=100000, ngram_range=(1,2))
x_train_features = count_vec.fit_transform(text_topic)

# Preparing the labels
y_train = categories

# Load the user data
user_data_df = pd.read_csv('./user_data.csv')
num_users = user_data_df['user_id'].nunique()
file_path = './telugu_news_translated.csv'
articles_df = pd.read_csv(file_path)
num_articles = articles_df.shape[0]

# Function to convert stringified lists to actual lists
def parse_list_string(list_string):
    try:
        return ast.literal_eval(list_string)
    except:
        return []


def get_article_index(sno, articles_df):
    indices = articles_df.index[articles_df['SNo'] == int(sno)].tolist()
    return indices[0] if indices else None

user_article_matrix = np.zeros((num_users, num_articles))
user_similarity_matrix = None
article_similarity_matrix = None

def create_matrix_after_enteringdata():
    global user_article_matrix
    global user_similarity_matrix
    global num_users
    global article_similarity_matrix
    global user_data_df

    user_data_df = pd.read_csv('./user_data.csv')


    user_data_df['liked_articles_index'] = user_data_df['liked_articles_sno'].apply(
        lambda x: [get_article_index(sno, articles_df) for sno in parse_list_string(x) if str(sno).isdigit() and get_article_index(sno, articles_df) is not None]
    )

    num_users = user_data_df['user_id'].nunique()

    # Resize the user_article_matrix if necessary
    if user_article_matrix.shape[0] != num_users:
        user_article_matrix = np.zeros((num_users, num_articles))

    # Populate the matrix with random values for both liked and unliked articles
    for index, row in user_data_df.iterrows():
        user_idx = index
        liked_indices = set(row['liked_articles_index'])
        for article_idx in range(num_articles):
            if article_idx in liked_indices:
                # Assign a higher random value for liked articles (e.g., between 0.5 and 1.0)
                user_article_matrix[user_idx, article_idx] = np.random.uniform(0.5, 1.0)
            else:
                # Assign a lower random value for unliked articles (e.g., between 0.1 and 0.3)
                user_article_matrix[user_idx, article_idx] = np.random.uniform(0.1, 0.3)

    # Normalize interactions
    user_article_matrix = normalize(user_article_matrix, norm='max', axis=1) * 4

    # Compute the cosine similarity matrices
    user_similarity_matrix = cosine_similarity(user_article_matrix)
    article_similarity_matrix = cosine_similarity(user_article_matrix.T)

    return num_users

def generate_ngrams(tokens, n=2):
    """Generate ngrams from tokens."""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def custom_analyzer_with_ngrams(text, n=2):
    """Tokenize text and generate n-grams."""
    words = indic_tokenize.trivial_tokenize(text)
    ngrams = generate_ngrams(words, n)
    return words + ngrams  # Return unigrams and bigrams

# Content based Recommendation
def get_recommendations_by_index(article_index, df_content, top_n=5):
    # article_index = find_article_index(article_id, df_content)
    # if article_index is None:
    #     print(article_id,"get_recommendations_by_index")
    #     return "Article not found", []
    # Compute tfidf with both unigrams and bigrams
    vect = TfidfVectorizer(analyzer=custom_analyzer_with_ngrams, ngram_range=(1,2), max_df=0.85, min_df=0.05)
    count_matrix = vect.fit_transform(df_content.body_processed.values)
    
    # Get the tf-idf vector for the specified article
    article_vector = count_matrix[article_index]

    # Compute the cosine similarity matrix for the specified article
    cosine_sim = linear_kernel(article_vector, count_matrix).flatten()
    
    # Get the similarity scores
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort by similarity
    sim_scores = sim_scores[1:top_n+1]  # Skip the first one as it is the article itself
    similar_article_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    # Retrieve the similar articles and their scores
    similar_articles = df_content['body_processed'].iloc[similar_article_indices]
    return similar_articles, scores

def recommend_articles_user_user_with_scores(user_index, df_content, top_n=5):
    # # Check if the user index is valid
    # if user_index not in user_data_df.index:
    #     return [], []

    user_category = user_data_df.loc[user_index, 'category']

    # Find users who like the same category
    same_category_user_indices = user_data_df[user_data_df['category'] == user_category].index.tolist()

    # Aggregate the article preferences of these users
    aggregated_article_scores = user_article_matrix[same_category_user_indices].sum(axis=0)

    # Sort articles by aggregated scores
    sorted_article_indices = np.argsort(aggregated_article_scores)[::-1]

    # Filter out articles the target user has already interacted with
    recommended_article_indices = [idx for idx in sorted_article_indices if user_article_matrix[user_index, idx] < 0.5][:top_n]

    # Retrieve the recommended articles and their scores
    recommended_articles = df_content.iloc[recommended_article_indices]
    scores = aggregated_article_scores[recommended_article_indices]

    return recommended_articles, scores



# Function to recommend similar articles for a given article using Item-Item Collaborative Filtering
def recommend_similar_articles_with_scores(article_index, df_content, top_n=5):
    # article_index = find_article_index(article_id, df_content)
    # if article_index is None:
    #     return "Article not found", []
    # Compute similarity scores
    similar_articles_scores = article_similarity_matrix[article_index]
    
    # Get top N similar articles, excluding the article itself
    similar_articles_indices = np.argsort(similar_articles_scores)[::-1][1:top_n+1]
    top_similar_scores = similar_articles_scores[similar_articles_indices]
    
    # Retrieve the recommended articles and their similarity scores
    recommended_articles = df_content.iloc[similar_articles_indices]
    return recommended_articles, top_similar_scores

from scipy.linalg import svd




def recommend_articles_svd_with_scores(user_index, df_content, top_n=5):
    # Perform SVD on the user-article matrix
    U, sigma, Vt = svd(user_article_matrix, full_matrices=False)

    # Reconstruct the user-article interaction matrix
    reconstructed_matrix = np.dot(U, np.dot(np.diag(sigma), Vt))

    # # Ensure the user index is valid
    # if user_index < 0 or user_index >= reconstructed_matrix.shape[0]:
    #     return "User index out of bounds", []

    # Extract the user's predicted ratings
    user_ratings = reconstructed_matrix[user_index]

    # Normalize and sort the ratings
    max_rating = max(user_ratings.max(), 1)  # Avoid division by zero
    normalized_scores = user_ratings / max_rating
    sorted_indices = np.argsort(normalized_scores)[::-1]

    # Select the top N articles, excluding already liked (interacted) articles
    recommended_indices = [idx for idx in sorted_indices if user_article_matrix[user_index, idx] == 0][:top_n]

    # Retrieve the recommended articles and their scores
    recommended_articles = df_content.iloc[recommended_indices]
    scores = normalized_scores[recommended_indices]

    return recommended_articles, scores



def hybrid_con_mat_recomm(article_sno, user_id, df_content, user_data_df, top_n=5, weight_content_based=0.6, weight_matrix_factorization=0.4):
    x=create_matrix_after_enteringdata()
    print(x)
    article_index = find_article_index(article_sno, df_content)
    if article_index is None:
        return "Article not found", []
    user_index = find_user_index(user_id, user_data_df)
    if user_index is None:
        return "User not found", []
    
    print(article_sno,user_id)
    cb_recommendations, cb_scores = get_recommendations_by_index(article_index, df_content, top_n)
    print("cb",cb_recommendations,cb_scores)
    mf_recommendations, mf_scores = recommend_articles_svd_with_scores(user_index, df_content, top_n)
    print("mf",mf_recommendations,mf_scores)

    combined_scores = {}
    for idx, score in zip(cb_recommendations.index, cb_scores):
        body = df_content.loc[idx, 'body_processed']
        combined_scores[body] = combined_scores.get(body, 0) + score * weight_content_based
    for idx, score in zip(mf_recommendations.index, mf_scores):
        body = df_content.loc[idx, 'body_processed']
        combined_scores[body] = combined_scores.get(body, 0) + score * weight_matrix_factorization

    sorted_articles = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_articles = [(article, score) for article, score in sorted_articles]

    return top_articles

def hybrid_con_user_recomm(article_sno, user_id, df_content, user_data_df, top_n=5, weight_content_based=0.1, weight_user_user=0.9):
    x=create_matrix_after_enteringdata()
    article_index = find_article_index(article_sno, df_content)
    if article_index is None:
        return "Article not found", []
    user_index = find_user_index(user_id, user_data_df)
    if user_index is None:
        return "User not found", []

    cb_recommendations, cb_scores = get_recommendations_by_index(article_index, df_content, top_n)
    print(cb_recommendations, cb_scores)
    uu_recommendations, uu_scores = recommend_articles_user_user_with_scores(user_index, df_content, top_n)
    print(uu_recommendations, uu_scores)

    combined_scores = {}
    for idx, score in zip(cb_recommendations.index, cb_scores):
        body = df_content.loc[idx, 'body_processed']
        combined_scores[body] = combined_scores.get(body, 0) + score * weight_content_based
    for idx, score in zip(uu_recommendations.index, uu_scores):
        body = df_content.loc[idx, 'body_processed']
        combined_scores[body] = combined_scores.get(body, 0) + score * weight_user_user

    sorted_articles = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_articles = sorted_articles

    return top_articles

def hybrid_item_user_recomm(article_sno, user_id, df_content, user_data_df, top_n=5, weight_item_item=0.7, weight_user_user=0.3):
    x=create_matrix_after_enteringdata()
    article_index = find_article_index(article_sno, df_content)
    if article_index is None:
        return "Article not found", []
    user_index = find_user_index(user_id, user_data_df)
    if user_index is None:
        return "User not found", []

    ii_recommendations, ii_scores = recommend_similar_articles_with_scores(article_index, df_content, top_n)
    uu_recommendations, uu_scores = recommend_articles_user_user_with_scores(user_index, df_content, top_n)

    combined_scores = {}
    for idx, score in zip(ii_recommendations.index, ii_scores):
        body = df_content.loc[idx, 'body_processed']
        combined_scores[body] = combined_scores.get(body, 0) + score * weight_item_item
    for idx, score in zip(uu_recommendations.index, uu_scores):
        body = df_content.loc[idx, 'body_processed']
        combined_scores[body] = combined_scores.get(body, 0) + score * weight_user_user

    sorted_articles = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_articles = [(article, score) for article, score in sorted_articles]

    return top_articles

def hybrid_item_mat_recomm(article_sno, user_id, df_content, user_data_df, top_n=5, weight_item_item=0.3, weight_matrix_factorization=0.7):
    x=create_matrix_after_enteringdata()
    article_index = find_article_index(article_sno, df_content)
    if article_index is None:
        return "Article not found", []
    user_index = find_user_index(user_id, user_data_df)
    if user_index is None:
        return "User not found", []

    ii_recommendations, ii_scores = recommend_similar_articles_with_scores(article_index, df_content, top_n)
    mf_recommendations, mf_scores = recommend_articles_svd_with_scores(user_index, df_content, top_n)

    combined_scores = {}
    for idx, score in zip(ii_recommendations.index, ii_scores):
        body = df_content.loc[idx, 'body_processed']
        combined_scores[body] = combined_scores.get(body, 0) + score * weight_item_item
    for idx, score in zip(mf_recommendations.index, mf_scores):
        body = df_content.loc[idx, 'body_processed']
        combined_scores[body] = combined_scores.get(body, 0) + score * weight_matrix_factorization

    sorted_articles = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_articles = [(article, score) for article, score in sorted_articles]

    return top_articles

