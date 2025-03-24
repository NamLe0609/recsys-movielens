# Avoid warning flooding the console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Default imports
import sys
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sps
from libreco.algorithms import NCF
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

def bfm_recommend(
    bfm_model, user_id, df_rating, ohe, movie_info_ohe, movie_genre_mle, movie_info, n
):
    unrated_movies = np.setdiff1d(
        df_rating["movieId"].unique(),
        df_rating[df_rating["userId"] == user_id]["movieId"].unique(),
    )

    X_unrated = ohe.transform(
        np.hstack(
            [np.full((len(unrated_movies), 1), user_id), unrated_movies.reshape(-1, 1)]
        )
    )
    X_side_info = movie_info_ohe.transform(
        movie_info.reindex(unrated_movies).drop(columns=["genres"])
    )
    X_genres = movie_genre_mle.transform(
        movie_info.genres.reindex(unrated_movies).str.split("|")
    )

    predictions = bfm_model.predict(sps.hstack([X_unrated, X_side_info, X_genres]))
    return unrated_movies[np.argsort(predictions)[-n:][::-1]]

def main():
    with open("model/best_bfm_model.pkl", "rb") as f:
        rs1 = pickle.load(f)
    with open("model/best_ncf_model/data_info.pkl", "rb") as f:
        data_info = pickle.load(f)
    rs2 = NCF.load(path="model/best_ncf_model", model_name="best_ncf_model", data_info=data_info)

    df_rating = pd.read_csv(f"ml-latest-small/ratings.csv")
    df_movie = pd.read_csv(f"ml-latest-small/movies.csv")
    df_movie["release_year"] = df_movie["title"].str.extract(r"\((\d{4})\)")
    df_movie["title"] = df_movie["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
    df_movie["release_year"] = df_movie["release_year"].astype(str)

    user_ids = df_rating['userId'].unique()
    FEATURE_COLUMNS = ["userId", "movieId"]
    ohe = OneHotEncoder(handle_unknown="ignore")
    ohe.fit(df_rating[FEATURE_COLUMNS])

    movie_info = df_movie.set_index("movieId")[["release_year", "genres"]]
    movie_info_ohe = OneHotEncoder(handle_unknown="ignore").fit(
        movie_info[["release_year"]]
    )
    movie_genre_mle = MultiLabelBinarizer(sparse_output=True).fit(
        movie_info.genres.apply(lambda x: x.split("|"))
    )

    try:
        print("\nLogin: Waiting User ID.")
        user_id = int(input("Enter User ID: ").strip())
        if user_id not in user_ids:
            print("User ID not found. Recommending for cold start user (popular).")
    except ValueError:
        print("Please enter a valid User ID.")
        sys.exit(1)

    while True:
        print("\nSelect a Recommender System:")
        print("1. RS1 (Bayesian Factorization Machine)")
        print("2. RS2 (Neural Collaborative Filtering)")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == '3':
            print("Exiting...")
            sys.exit(0)
        
        if choice not in ['1', '2']:
            print("Invalid choice. Please try again.")
            continue

        model = int(choice)

        try:
            n = int(input("How many movies do you want recommended? ").strip())
        except ValueError:
            print("Please enter a valid number.")
            continue

        if (user_id in user_ids) and model == 1:
            recommendation = bfm_recommend(rs1, user_id, df_rating, ohe, movie_info_ohe, movie_genre_mle, movie_info, n)
        else:
            recommendation = rs2.recommend_user(user=user_id, n_rec=n, cold_start="average")[user_id]

        for i, movie in enumerate(recommendation):
            print(f"    {i+1}. {df_movie[df_movie['movieId'] == movie]['title'].values[0]}")

if __name__ == "__main__":
    main()
