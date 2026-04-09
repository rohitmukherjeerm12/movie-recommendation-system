import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load the datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge both files into one table
movies = movies.merge(credits, on='title')

# Keep only the columns we need
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop rows with missing values
movies.dropna(inplace=True)

# This function converts JSON-like text into a simple list
# For example: [{"name": "Action"}, {"name": "Drama"}] → ["Action", "Drama"]
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

# Apply the function to genres and keywords columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# This function keeps only top 3 cast members
def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

# This function extracts only the director from crew
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Convert overview from a sentence into a list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces from names so "Sam Mendes" becomes "SamMendes"
# This prevents confusion between different people with same first name
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Combine all columns into one single "tags" column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a clean final table with just title and tags
new_df = movies[['movie_id', 'title', 'tags']]

# Convert tags list into a single string
new_df = new_df.copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

print(new_df.head())
print("Data processing done!")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Convert text tags into numbers (vectors)
# max_features=5000 means we only use top 5000 most common words
# stop_words='english' removes useless words like "the", "is", "and"
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Calculate similarity between every pair of movies
# This creates a big table — each movie compared to every other movie
similarity = cosine_similarity(vectors)

# The actual recommendation function
# Takes a movie name → finds its position → sorts by similarity → returns top 5
def recommend(movie):
    # Find the index (row number) of the movie in our table
    movie_index = new_df[new_df['title'] == movie].index[0]
    
    # Get similarity scores of this movie with all others
    distances = similarity[movie_index]
    
    # Sort movies by similarity score (highest first)
    # [1:6] means skip the first result (that's the movie itself) and take next 5
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    # Print the recommended movies
    print(f"\nMovies similar to '{movie}':")
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# Test it!
recommend('Avatar')

import pickle

# Save the movies list and similarity matrix to files
# Think of pickle like a "freeze" function — it saves Python objects to a file
pickle.dump(new_df.to_dict(), open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("Model saved successfully!")