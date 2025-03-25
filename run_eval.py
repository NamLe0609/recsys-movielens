# Avoid warning flooding the console
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore")
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

# Default imports
import pandas as pd
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import KFold

import scipy.sparse as sps
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn import metrics

import myfm

from libreco.data import random_split, DatasetFeat
from libreco.evaluation import evaluate
from libreco.algorithms import NCF

import optuna


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


def get_extended_features(
    X_train_base,
    X_test_base,
    df_train,
    df_test,
    incluse_variance=False,
    include_release_year=False,
    include_genres=False,
):
    X_train_ext = X_train_base
    X_test_ext = X_test_base
    group_shapes = None

    if incluse_variance:
        group_shapes = [len(group) for group in ohe.categories_]

    if include_release_year:
        X_train_release = movie_info_ohe.transform(
            movie_info.reindex(df_train.movieId).drop(columns=["genres"])
        )
        X_test_release = movie_info_ohe.transform(
            movie_info.reindex(df_test.movieId).drop(columns=["genres"])
        )
        X_train_ext = sps.hstack([X_train_ext, X_train_release])
        X_test_ext = sps.hstack([X_test_ext, X_test_release])
        group_shapes.extend([len(group) for group in movie_info_ohe.categories_])

    if include_genres:
        X_train_genres = movie_genre_mle.transform(
            movie_info.genres.reindex(df_train.movieId).apply(lambda x: x.split("|"))
        )
        X_test_genres = movie_genre_mle.transform(
            movie_info.genres.reindex(df_test.movieId).apply(lambda x: x.split("|"))
        )
        X_train_ext = sps.hstack([X_train_ext, X_train_genres])
        X_test_ext = sps.hstack([X_test_ext, X_test_genres])
        group_shapes.append(len(movie_genre_mle.classes_))

    return X_train_ext, X_test_ext, group_shapes


def train_predict_fm(
    df,
    include_variance=False,
    include_release_year=False,
    include_genres=False,
    fm_rank=10,
    n_splits=5,
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []
    mae_scores = []

    for train_index, test_index in kf.split(df):
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]

        X_train_base = ohe.transform(df_train[FEATURE_COLUMNS])
        X_test_base = ohe.transform(df_test[FEATURE_COLUMNS])
        y_train = df_train.rating.values
        y_test = df_test.rating.values

        X_train_ext, X_test_ext, group_shapes = get_extended_features(
            X_train_base,
            X_test_base,
            df_train,
            df_test,
            incluse_variance=include_variance,
            include_release_year=include_release_year,
            include_genres=include_genres,
        )

        model = myfm.MyFMRegressor(rank=fm_rank, random_seed=42)
        model.fit(
            X_train_ext,
            y_train,
            n_iter=200,
            n_kept_samples=200,
            group_shapes=group_shapes,
        )
        prediction = model.predict(X_test_ext)
        rmse = np.sqrt(((y_test - prediction) ** 2).mean())
        mae = np.abs(y_test - prediction).mean()

        rmse_scores.append(rmse)
        mae_scores.append(mae)

    mean_rmse = np.mean(rmse_scores)
    mean_mae = np.mean(mae_scores)

    return mean_rmse, mean_mae


def novelty(recommendations, interactions, item_popularity):
    novelty_scores = [
        -np.log2(item_popularity.get(item, 1) / interactions)
        for item in recommendations
    ]
    return np.mean(novelty_scores)


data_path = "ml-latest-small/"
df_rating = pd.read_csv(f"{data_path}ratings.csv")
df_movie = pd.read_csv(f"{data_path}movies.csv")
df_movie["release_year"] = df_movie["title"].str.extract(r"\((\d{4})\)")
df_movie["title"] = df_movie["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
df_movie["release_year"] = df_movie["release_year"].astype(str)

print("Training and evaluating models... (All evaluation metrics shown in video are recorded during training)")
# RS2: Neural Collaborative Filtering: RMSE/MAE
mlb = MultiLabelBinarizer()
libreco_full = df_rating.merge(df_movie, how="left", on="movieId")
libreco_full = libreco_full.rename(
    columns={"userId": "user", "movieId": "item", "rating": "label"}
)
libreco_full["genres"] = libreco_full["genres"].str.split("|")
genre_encoded = pd.DataFrame(
    mlb.fit_transform(libreco_full["genres"]),
    columns=mlb.classes_,
    index=libreco_full.index,
)
libreco_full = pd.concat([libreco_full, genre_encoded], axis=1).drop(
    columns=["genres", "timestamp"]
)

train_data, eval_data, test_data = random_split(
    libreco_full, multi_ratios=[0.8, 0.1, 0.1], seed=42
)

sparse_col = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
dense_col = ["release_year"]
user_col = []
item_col = sparse_col + dense_col

train_data, data_info = DatasetFeat.build_trainset(
    train_data,
    user_col=user_col,
    item_col=item_col,
    sparse_col=sparse_col,
    dense_col=dense_col,
)
eval_data = DatasetFeat.build_testset(eval_data)
test_data = DatasetFeat.build_testset(test_data)

study = optuna.load_study(
    study_name="NCF_RMSE_Optimization", storage="sqlite:///ncf_study.db"
)

best_trial = study.best_trial
ncf_model = NCF(
    task="rating",
    data_info=data_info,
    embed_size=best_trial.params["embed_size"],
    n_epochs=best_trial.params["n_epochs"],
    lr=best_trial.params["lr"],
    batch_size=best_trial.params["batch_size"],
    hidden_units=best_trial.params["hidden_units"],
    dropout_rate=best_trial.params["dropout_rate"],
    reg=best_trial.params["reg"],
    sampler="unconsumed",
    use_bn=True,
    seed=42,
)
ncf_model.fit(train_data, neg_sampling=False, eval_data=eval_data, verbose=0)

ncf_result = evaluate(
    model=ncf_model,
    data=test_data,
    neg_sampling=False,
    metrics=["rmse", "mae"],
)

train_data, eval_data, test_data = random_split(
    libreco_full, multi_ratios=[0.8, 0.1, 0.1], seed=42
)

item_popularity = libreco_full.groupby("item")["user"].nunique().to_dict()
interactions = len(libreco_full)

user_novelties = []
for user_id in libreco_full["user"].unique():
    user_novelties.append(
        novelty(
            ncf_model.recommend_user(user=user_id, n_rec=10)[user_id],
            interactions,
            item_popularity,
        )
    )
overall_novelty = np.mean(user_novelties)

# RS1: Bayesian Factorization Machine: RMSE/MAE
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

X_train = ohe.fit_transform(df_rating[FEATURE_COLUMNS])
y_train = df_rating.rating.values

output = train_predict_fm(
    df=df_rating,
    include_variance=True,
    include_release_year=True,
    include_genres=True,
    fm_rank=20,
    n_splits=5,
)

X_train_extended = sps.hstack(
    [
        X_train,
        movie_info_ohe.transform(
            movie_info.reindex(df_rating.movieId).drop(columns=["genres"])
        ),
        movie_genre_mle.transform(
            movie_info.genres.reindex(df_rating.movieId).apply(lambda x: x.split("|"))
        ),
    ]
)

group_shapes_extended = (
    [len(group) for group in ohe.categories_]
    + [len(group) for group in movie_info_ohe.categories_]
    + [len(movie_genre_mle.classes_)]
)

best_bfm = myfm.MyFMRegressor(
    rank=20,
    random_seed=42,
)
best_bfm.fit(
    X_train_extended,
    y_train,
    n_iter=200,
    n_kept_samples=200,
    group_shapes=group_shapes_extended,
)

item_popularity = df_rating.groupby("movieId")["userId"].nunique().to_dict()
interactions = len(df_rating)

print(
    f"RS2 (Neural Colaborative Filtering): RMSE: {ncf_result['rmse']:.4f}, MAE: {ncf_result['mae']:.4f}"
)
print(f"RS2 (Neural Colaborative Filtering): Novelty: {overall_novelty:.4f}")
print(
    f"RS1 (Bayesian Factorization Machine): RMSE: {output[0]:.4f}, MAE: {output[1]:.4f}"
)
print("Calculating novelty. This may take a while... (approx. 10 minutes)")
user_novelties = []
for user_id in df_rating["userId"].unique():
    user_novelties.append(
        novelty(
            bfm_recommend(
                best_bfm,
                user_id,
                df_rating,
                ohe,
                movie_info_ohe,
                movie_genre_mle,
                movie_info,
                10,
            ),
            interactions,
            item_popularity,
        )
    )
overall_novelty = np.mean(user_novelties)

print(f"RS1 (Neural Colaborative Filtering): Novelty: {overall_novelty:.4f}")
