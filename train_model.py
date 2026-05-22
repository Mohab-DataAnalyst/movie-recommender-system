from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


BASE_DIR = Path(__file__).resolve().parent
MOVIES_PATH = BASE_DIR / "movies.csv"
RATINGS_PATH = BASE_DIR / "ratings.csv"
ARTIFACT_PATH = BASE_DIR / "recommender_artifacts.pkl"


def main():
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    rated_movie_ids = ratings["movieId"].unique().tolist()
    user_ids = ratings["userId"].unique().tolist()

    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(rated_movie_ids)}
    movie_idx_to_id = {idx: movie_id for movie_id, idx in movie_id_to_idx.items()}
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    user_idx_to_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}

    user_idx = ratings["userId"].map(user_id_to_idx)
    movie_idx = ratings["movieId"].map(movie_id_to_idx)
    ratings_matrix = csr_matrix(
        (ratings["rating"], (user_idx, movie_idx)),
        shape=(len(user_ids), len(rated_movie_ids)),
    )

    movies = movies.reset_index(drop=True)
    genres = sorted(
        {
            genre
            for genre_list in movies["genres"].str.split("|")
            for genre in genre_list
            if genre != "(no genres listed)"
        }
    )
    movie_genres = movies["genres"].str.get_dummies(sep="|")
    movie_genres = movie_genres.reindex(columns=genres, fill_value=0)
    movie_genres_matrix = csr_matrix(movie_genres.values)

    n_components = min(20, min(ratings_matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_matrix = svd.fit_transform(ratings_matrix)
    movie_matrix = svd.components_.T

    movie_id_to_title = dict(zip(movies["movieId"], movies["title"]))
    movie_title_to_id = dict(zip(movies["title"], movies["movieId"]))

    artifacts = {
        "movies": movies,
        "titles": movies["title"].tolist(),
        "user_ids": user_ids,
        "ratings_matrix": ratings_matrix,
        "movie_genres_matrix": movie_genres_matrix,
        "user_matrix": user_matrix,
        "movie_matrix": movie_matrix,
        "movie_id_to_title": movie_id_to_title,
        "movie_title_to_id": movie_title_to_id,
        "movie_id_to_idx": movie_id_to_idx,
        "movie_idx_to_id": movie_idx_to_id,
        "user_id_to_idx": user_id_to_idx,
        "user_idx_to_id": user_idx_to_id,
    }

    joblib.dump(artifacts, ARTIFACT_PATH)

    print(f"Users: {len(user_ids)}")
    print(f"Movies: {len(movies)}")
    print(f"Ratings: {len(ratings)}")
    print(f"Saved artifacts to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
