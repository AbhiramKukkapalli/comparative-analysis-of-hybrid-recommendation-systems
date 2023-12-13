import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def home():
    st.title("Dashboard")

    # Load the data
    df = pd.read_csv('user_ratings.csv')

    # Create two columns for the first two graphs
    col1, col2 = st.columns(2)

    # 1. Average Rating Per Recommendation System
    with col1:
        st.subheader("Average Rating Per System")
        st.write("This graph shows the average rating for each recommendation system.")
        avg_ratings = df.groupby('system')['rating'].mean().sort_values()
        fig, ax = plt.subplots()
        avg_ratings.plot(kind='barh', ax=ax)
        ax.set_xlabel('Average Rating')
        st.pyplot(fig)

    # 2. Ratings Distribution for Each System
    with col2:
        st.subheader("Ratings Distribution per System")
        st.write("Box plot showing the distribution of ratings for each recommendation system.")
        fig, ax = plt.subplots()
        sns.boxplot(x='rating', y='system', data=df, ax=ax)
        ax.set_xlabel('Rating')
        ax.set_ylabel('System')
        st.pyplot(fig)

    # Create two columns for the next two graphs
    col3, col4 = st.columns(2)

    # 3. Comparison of Ratings Across Categories
    with col3:
        st.subheader("Comparison Across User Selected Categories")
        st.write("Average ratings for each system across different categories selected by users.")
        fig, ax = plt.subplots()
        sns.barplot(x='selected_category', y='rating', hue='system', data=df, ax=ax)
        ax.set_xlabel('Selected Category')
        ax.set_ylabel('Average Rating')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # 1. User Satisfaction by Article Category
    with col4:
        st.subheader("User Satisfaction by Article Category")
        st.write("Evaluates how users rate each system across different article categories.")
        fig, ax = plt.subplots()
        sns.barplot(x='category', y='rating', hue='system', data=df, ax=ax)
        ax.set_xlabel('Article Category')
        ax.set_ylabel('Average Rating')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # 2. User Preference Alignment
    st.subheader("User Preference Alignment")
    st.write("Comparison of average ratings in context of users' preferred content categories.")
    user_pref_df = df.groupby(['selected_category', 'system'])['rating'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='selected_category', y='rating', hue='system', data=user_pref_df, ax=ax)
    ax.set_xlabel('User Selected Category')
    ax.set_ylabel('Average Rating')
    plt.xticks(rotation=45)
    st.pyplot(fig)



def add_user_data():
    # Set the session state to indicate redirection to Add_data.py
    st.session_state.current_page = "Add User Data"
    st.experimental_rerun()

def display_user_data():
    st.title("Display User Data")
    # Logic to display user data and ratings

# Main App
st.sidebar.title('Navigation')
choice = st.sidebar.radio("Go to", ['Home', 'Add User Data', 'Display User Data'])

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Page navigation logic
if st.session_state.current_page == "Home":
    home()
elif st.session_state.current_page == "Add User Data":
    # Redirect to Add_data.py
    exec(open("Add_data.py").read())
elif st.session_state.current_page == "Display User Data":
    display_user_data()
else:
    # Default case
    home()

# Update session state based on sidebar choice
if choice != st.session_state.current_page:
    st.session_state.current_page = choice
    st.experimental_rerun()
