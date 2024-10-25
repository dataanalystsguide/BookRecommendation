import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Display logo
image = Image.open('logo.png')
st.image(image)
st.title("-- Book Recommender System 📔--")

# Introduction text
st.markdown("Hey there! 📚 Excited about finding your next great read? BookHub is your personalized book buddy, crafting recommendations just for you based on what you love. We use collaborative filtering techniques to offer up a handpicked selection of book titles to make your reading journey even more delightful! Ready to dive in and discover your next favorite?")

# User input for name
st.subheader("Enter your Name")
user_name = st.text_input("")

# Load the dataset
Item_based = pd.read_csv('https://github.com/aasimshaikh98/BookHub/blob/master/item_based.csv?raw=true')

# Main functionality
if user_name:
    st.markdown("### ------------ Welcome to BookHub's Overall Library -----------")
    st.markdown("Step into our Overall Library section, where an array of captivating reads awaits! 📚")
    st.subheader("Which Book are you looking for?")
    
    # Book search input
    book = st.text_input("##### Enter something related to the book you're looking for!")
    
    if book:
        # Vectorizer setup
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(Item_based['altered_title'])

        # Similar books function
        def similar_books(book_name: str):
            book_name = re.sub('[^a-zA-Z0-9]', ' ', book_name.lower()).strip()
            book_vector = vectorizer.transform([book_name])
            similarity = cosine_similarity(book_vector, tfidf).flatten()
            similar_book_id = np.flip(np.argpartition(similarity, -20)[-20:])
            similar_books_df = pd.DataFrame(columns=['Book-Title', 'Mean-Rating', 'Book-Author'])

            for i in similar_book_id:
                similar_books_df = pd.concat([similar_books_df, pd.DataFrame(Item_based.iloc[i][['Book-Title', 'Mean-Rating', 'Book-Author']]).transpose()])

            similar_books_df.reset_index(drop=True, inplace=True)
            similar_books_df.drop_duplicates(inplace=True)
            similar_books_df['Book & Author'] = similar_books_df['Book-Title'] + ' --- ' + similar_books_df['Book-Author']
            return similar_books_df

        # Display recommendations
        final = similar_books(book)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader(f"List of recommendations for {user_name} 📕📓📕")

        for i in range(len(final)):
            st.write(f"📓 {final.iloc[i]['Book-Title']} 📓 --- ⭐ {final.iloc[i]['Mean-Rating']} ⭐ --- ✍🏻 {final.iloc[i]['Book-Author']} ✍🏻")

        # Filter by rating
        st.subheader("Filter by ⭐ Ratings")
        rating = st.slider(" ", 0, 5)
        if rating:
            st.subheader(f"Here are recommendations filtered by ratings 📕📓📕")
            for i in range(len(final)):
                if final.iloc[i]['Mean-Rating'] == rating:
                    st.write(f"📓 {final.iloc[i]['Book-Title']} 📓 --- ⭐ {final.iloc[i]['Mean-Rating']} ⭐ --- ✍🏻 {final.iloc[i]['Book-Author']} ✍🏻")
