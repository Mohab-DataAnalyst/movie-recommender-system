from pathlib import Path
from difflib import get_close_matches

import joblib
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


ARTIFACT_PATH = Path(__file__).resolve().parent / "recommender_artifacts.pkl"


@st.cache_resource
def load_artifacts():
    return joblib.load(ARTIFACT_PATH)


def find_title(query, titles):
    query_lower = query.lower()
    exact_matches = [title for title in titles if title.lower() == query_lower]
    if exact_matches:
        return exact_matches[0]

    contains_matches = [title for title in titles if query_lower in title.lower()]
    if contains_matches:
        return contains_matches[0]

    matches = get_close_matches(query, titles, n=1, cutoff=0.35)
    return matches[0] if matches else None


def content_recommendations(title, n, artifacts):
    movies = artifacts["movies"]
    movie_genres_matrix = artifacts["movie_genres_matrix"]
    idx = int(movies.index[movies["title"] == title][0])
    sim_scores = cosine_similarity(movie_genres_matrix[idx], movie_genres_matrix).ravel()
    top_indices = np.argsort(sim_scores)[::-1]
    top_indices = [i for i in top_indices if i != idx][:n]
    return movies.iloc[top_indices]["title"].tolist()


def collaborative_recommendations(title, n, artifacts):
    movie_title_to_id = artifacts["movie_title_to_id"]
    movie_id_to_idx = artifacts["movie_id_to_idx"]
    movie_idx_to_id = artifacts["movie_idx_to_id"]
    movie_id_to_title = artifacts["movie_id_to_title"]
    ratings_matrix = artifacts["ratings_matrix"]

    movie_id = movie_title_to_id.get(title)
    if movie_id not in movie_id_to_idx:
        return content_recommendations(title, n, artifacts)

    movie_idx = movie_id_to_idx[movie_id]
    movie_matrix = ratings_matrix.T
    k = min(n + 1, movie_matrix.shape[0])
    model = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="cosine")
    model.fit(movie_matrix)
    neighbors = model.kneighbors(movie_matrix[movie_idx], return_distance=False).ravel()

    recommendations = []
    for neighbor_idx in neighbors:
        neighbor_movie_id = movie_idx_to_id[int(neighbor_idx)]
        neighbor_title = movie_id_to_title[neighbor_movie_id]
        if neighbor_title != title:
            recommendations.append(neighbor_title)
        if len(recommendations) == n:
            break
    return recommendations


def matrix_factorization_recommendations(user_id, n, artifacts):
    user_id_to_idx = artifacts["user_id_to_idx"]
    movie_idx_to_id = artifacts["movie_idx_to_id"]
    movie_id_to_title = artifacts["movie_id_to_title"]
    ratings_matrix = artifacts["ratings_matrix"]
    user_matrix = artifacts["user_matrix"]
    movie_matrix = artifacts["movie_matrix"]

    user_idx = user_id_to_idx[user_id]
    scores = user_matrix[user_idx].dot(movie_matrix.T)
    rated_movie_indices = ratings_matrix[user_idx].nonzero()[1]
    scores[rated_movie_indices] = -np.inf
    top_movie_indices = np.argsort(scores)[-n:][::-1]
    return [movie_id_to_title[movie_idx_to_id[int(i)]] for i in top_movie_indices]


st.set_page_config(page_title="Movie Recommender")
st.title("Movie Recommender")
st.write("Choose a strategy and get movie recommendations from ratings and genre data.")

if not ARTIFACT_PATH.exists():
    st.error("Artifacts not found. Run `python train_model.py` first.")
    st.stop()

artifacts = load_artifacts()
titles = artifacts["titles"]
user_ids = artifacts["user_ids"]

with st.form("recommendation_form"):
    strategy = st.selectbox(
        "Recommendation strategy",
        ["auto", "content", "collaborative", "matrix_factorization"],
    )
    movie_title = st.text_input("Movie title", value="Toy Story")
    user_id = st.number_input("User ID", min_value=1, value=int(user_ids[0]), step=1)
    n_recommendations = st.slider("Number of recommendations", 1, 20, 5)
    submitted = st.form_submit_button("Recommend")

if submitted:
    matched_title = None
    recommendations = []

    if strategy == "matrix_factorization":
        if user_id not in artifacts["user_id_to_idx"]:
            st.error(f"User ID {user_id} was not found in the ratings data.")
            st.stop()
        recommendations = matrix_factorization_recommendations(
            user_id,
            n_recommendations,
            artifacts,
        )
    else:
        matched_title = find_title(movie_title, titles)
        if matched_title is None:
            st.error("No close movie title match found.")
            st.stop()

        if strategy == "content":
            recommendations = content_recommendations(
                matched_title,
                n_recommendations,
                artifacts,
            )
        elif strategy == "collaborative":
            recommendations = collaborative_recommendations(
                matched_title,
                n_recommendations,
                artifacts,
            )
        elif strategy == "auto" and user_id in artifacts["user_id_to_idx"]:
            recommendations = matrix_factorization_recommendations(
                user_id,
                n_recommendations,
                artifacts,
            )
        else:
            recommendations = content_recommendations(
                matched_title,
                n_recommendations,
                artifacts,
            )

    if matched_title:
        st.caption(f"Matched movie: {matched_title}")
    st.subheader("Recommendations")
    for i, title in enumerate(recommendations, start=1):
        st.write(f"{i}. {title}")
