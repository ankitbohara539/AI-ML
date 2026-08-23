import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ====1. load data===========

movies = pd.read_csv("ml-latest-small/movies.csv")
print("movies dataset shape:")
print(movies.shape)

print("\n first 5 movies:")
print(movies.head())


# ======== 2. Handle the missing values in the 'genres' column ========

movies['genres'] = movies['genres'].fillna('')  # Fill missing values with an empty string

# ======== 3. Create the content features ========

# r = regex pattern to match words, \b is a word boundary, \w+ matches one or more word characters, and (?u) makes it Unicode-aware. This will help in tokenizing the genres correctly.

vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b")

genre_metrtix = vectorizer.fit_transform(movies["genres"])

print("\nGenre feature matrix shape:")
print(genre_metrtix.shape)

# ======== 4. Calculate the simiilarity ========

similarity_matrix = cosine_similarity(genre_metrtix)
print("\n similarity matrix shape:")
print(similarity_matrix.shape)


#========= 5. create a movie index mapping for easy lookup ========
movie_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()


# ========= 6. Create Recommendation Function ========
def recommend_movies(movie_title,number_of_recommendations=10):

    if movie_title not in movie_indices:
        print("\nMovie not found")
        return

    movie_index = movie_indices[movie_title]

    similarity_scores = list(
        enumerate(
            similarity_matrix[movie_index]
        )
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],reverse=True
    )

    similarity_scores = similarity_scores[
        1:number_of_recommendations + 1
    ]

    movie_indices_list = [
        index
        for index, score
        in similarity_scores
    ]

    recommendations = movies.iloc[
        movie_indices_list
    ][
        [
            "movieId",
            "title",
            "genres",
        ]
    ]

    return recommendations

#==========8. make recommendations==============

movie_title = "Toy Story (1995)"
recommendations = recommend_movies(movie_title)

print("\n=======CONTENT-BASED RECOMMENDATIONS")
print("\nBecause you liked:", movie_title)
print(recommendations.to_string(index=False))