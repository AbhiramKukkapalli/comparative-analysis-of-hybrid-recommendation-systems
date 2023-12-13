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
import os
from hybrid_recommendations import telugu_news_df,test_news_df
from hybrid_recommendations import hybrid_con_mat_recomm,hybrid_con_user_recomm,hybrid_item_user_recomm,hybrid_item_mat_recomm 

# Example data structure
user_data = {
    "user_id": [], 
    "category": [],
    "liked_articles_sno": [], 
    "timestamp": []
}



# Placeholder function to fetch articles based on a category
def get_all_articles():
    return articles_df.to_dict(orient='records')

def record_likes(user_id, category, liked_articles_sno):
    global user_data
    user_data["user_id"].append(user_id)
    user_data["category"].append(category)
    user_data["liked_articles_sno"].append(liked_articles_sno)
    user_data["timestamp"].append(datetime.now())

# Function to save ratings to a CSV file
def save_ratings_to_csv(user_id, selected_category, article_category, ratings, filename='user_ratings.csv'):
    # Check if the CSV file exists
    if not os.path.isfile(filename):
        # Create a new DataFrame and CSV file if it doesn't exist
        columns = ['user_id', 'system', 'rating', 'timestamp', 'category', 'selected_category']
        df = pd.DataFrame(columns=columns)
        df.to_csv(filename, index=False)

    # Prepare the data to be written
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = []
    for system, rating in ratings.items():
        data.append([user_id, system, rating, timestamp, article_category, selected_category])

    # Append the new ratings to the CSV file
    df = pd.DataFrame(data, columns=['user_id', 'system', 'rating', 'timestamp', 'category', 'selected_category'])
    df.to_csv(filename, mode='a', header=False, index=False)


def write_data_to_csv():
    global user_data
    df = pd.DataFrame(user_data)
    df.to_csv('user_data.csv', mode='a', header=False, index=False)


def get_hybrid_recommendations(selected_article_sno, user_id, df_content, user_data_df):
    recommendations = {
        'Content-Matrix': hybrid_con_mat_recomm(selected_article_sno, user_id, df_content, user_data_df),
        'Content-User': hybrid_con_user_recomm(selected_article_sno, user_id, df_content, user_data_df),
        'Item-User': hybrid_item_user_recomm(selected_article_sno, user_id, df_content, user_data_df),
        'Item-Matrix': hybrid_item_mat_recomm(selected_article_sno, user_id, df_content, user_data_df)
    }
    return recommendations

def add_user_data():
    st.title("Add User Data")
    user_id = None

     # Initialize a flag in session_state to track if likes are saved
    if 'likes_saved' not in st.session_state:
        st.session_state['likes_saved'] = False

    # Initialize a dictionary in session_state to track likes
    if 'likes' not in st.session_state:
        st.session_state['likes'] = {}

    # User selects a category
    categories = articles_df['topic'].unique()
    selected_category  = st.selectbox("Select a category of articles you like", categories)
    
    # Fetch articles based on the category
    articles = get_all_articles()

    
    # HTML and CSS for the scrollable grid container with JavaScript
    html_content = """
    <style>
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        gap: 10px;
        padding: 10px;
        overflow-y: auto;
        height: 500px;
    }
    .article-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        background-color: #f9f9f9;
        cursor: pointer;
    }
    .modal {
        display: none;
        position: fixed;
        z-index: 1;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgb(0,0,0);
        background-color: rgba(0,0,0,0.4);
        padding-top: 60px;
    }
    .modal-content {
        background-color: #fefefe;
        margin: 5% auto;
        padding: 20px;
        border: 1px solid #888;
        width: 80%;
    }
    .close {
        color: #aaa;
        float: right;
        font-size: 28px;
        font-weight: bold;
    }
    .close:hover,
    .close:focus {
        color: black;
        text-decoration: none;
        cursor: pointer;
    }
    </style>
    <div class="grid-container">
    """
    i=0
    # Adding articles to the grid with modal pop-up
    for article in articles:
        i+=1
        html_content += f"""
        <div class='article-box' onclick="document.getElementById('modal_{article['heading-translated']}').style.display = 'block';">
            <h4>{article['SNo']}</h4>
            <h5>{article['heading-translated']}</h5>
        </div>
        <div id="modal_{article['heading-translated']}" class="modal">
            <div class="modal-content">
                <span class="close" onclick="document.getElementById('modal_{article['heading-translated']}').style.display = 'none';">&times;</span>
                <p>{article['body']}</p>
                <label for="like_{article['heading-translated']}">Like this article</label>
            </div>
        </div>
        """

    html_content += "</div>"

    # Display the grid
    components.html(html_content, height=600, width=800)

    likes_input = st.text_input("SNo of Liked Articles from above (comma-separated)", value='', key='likes_input')
    
   # Use a button to save likes
    if st.button("Save my likes"):
        # Generate a unique user ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_number = random.randint(100, 9999)
        user_id = f"user_{timestamp}_{random_number}"
        st.session_state['user_id'] = user_id 
        liked_articles_sno = st.session_state['likes_input'].split(',')
        record_likes(user_id, selected_category, liked_articles_sno)
        write_data_to_csv()
        # Set the flag to True after saving likes
        st.session_state['likes_saved'] = True
        st.success(f"Your likes have been saved.")




   # Creating a dictionary to map article titles to their SNo
    article_title_to_sno = dict(zip(articles_df['heading-translated'], articles_df['SNo']))

    # User selects one article for detailed recommendations
    st.subheader("Select one article to get recommendations:")
    article_titles = articles_df['heading-translated'].tolist()
    selected_article_title = st.selectbox("Choose an article", article_titles)

    # Get the SNo for the selected article
    selected_article_sno = article_title_to_sno[selected_article_title]

    # Retrieve the category of the selected article
    # Assuming the column name for the category in your DataFrame is 'topic'
    selected_article_category = articles_df.loc[articles_df['SNo'] == selected_article_sno, 'topic'].values[0]

    # Display the selected article's category
    st.write(f"Selected Article Category: {selected_article_category}")

    
    
    # Inside your Streamlit app
    if st.button("Get Recommendations"):
        if st.session_state.get('likes_saved', False) and 'user_id' in st.session_state and selected_article_sno:
            user_id = st.session_state['user_id']
            # Store recommendations in session state
            st.session_state['recommendations'] = get_hybrid_recommendations(selected_article_sno, user_id, df_content, user_data_df)
            for system, recs in st.session_state['recommendations'].items():
                st.subheader(f"Recommendations from {system}")
                for rec_index, rec in enumerate(recs):
                    article_body = rec[0]
                    article_summary = article_body[:100] + "..."
                    with st.expander(f"Article {rec_index + 1} Summary: {article_summary}"):
                        st.write(article_body)
        else:
            st.error("Please save your likes and input user data before getting recommendations.")

   # Example usage in Streamlit
    # Example usage in Streamlit
    if 'recommendations' in st.session_state:
        ratings = {}
        for system in st.session_state['recommendations']:
            rating_key = f"rating_{system}_{user_id}"
            ratings[system] = st.slider(f"Rate the recommendations from {system}", 1, 5, key=rating_key)
        st.session_state['user_ratings'] = ratings

        if st.button("Save Ratings"):
            if 'user_ratings' in st.session_state and 'user_id' in st.session_state:
                # Assume 'selected_article_category' is the category of the selected article
                save_ratings_to_csv(st.session_state['user_id'], selected_category, selected_article_category, st.session_state['user_ratings'])
                st.success("Ratings have been saved successfully!")
            else:
                st.error("No ratings to save. Please rate the recommendations first.")








# In your main app function
def main():
    # Rest of your Streamlit app logic
    add_user_data()


if __name__ == "__main__":
    # Load the CSV file
    file_path = './telugu_news_translated.csv'
    articles_df = pd.read_csv(file_path)
    num_articles = articles_df.shape[0]
    df_content=test_news_df
    user_data_df = pd.read_csv('./user_data.csv')

    main()