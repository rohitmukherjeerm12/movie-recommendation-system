import streamlit as st
import pickle
import pandas as pd

# Load the saved model files
movies_dict = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Recommendation function (same logic as before)
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended = []
    for i in movies_list:
        recommended.append(movies.iloc[i[0]].title)
    return recommended

# ---------- WEB PAGE DESIGN ----------

# Page title and description
st.title('🎬 Movie Recommendation System')
st.markdown('Select a movie and get 5 similar movie recommendations instantly!')

# Dropdown to select a movie
selected_movie = st.selectbox(
    'Type or select a movie:',
    movies['title'].values
)

# Button to trigger recommendation
if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    
    st.subheader('Movies you might like:')
    for i, movie in enumerate(recommendations):
        st.write(f"{i+1}. {movie}")