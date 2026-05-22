from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


ARTIFACT_PATH = Path(__file__).resolve().parent / "recommender_artifacts.pkl"


class RecommendationRequest(BaseModel):
    strategy: Literal["auto", "content", "collaborative", "matrix_factorization"] = "auto"
    movie_title: str | None = Field(None, examples=["Toy Story"])
    user_id: int | None = Field(None, ge=1, examples=[25])
    n_recommendations: int = Field(5, ge=1, le=20)


app = FastAPI(
    title="Movie Recommender API",
    description="A simple FastAPI demo for movie recommendations.",
    version="1.0.0",
)


def load_artifacts():
    if not ARTIFACT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Artifacts not found. Run `python train_model.py` first.",
        )
    return joblib.load(ARTIFACT_PATH)


def find_title(query, titles):
    if not query:
        raise HTTPException(status_code=400, detail="movie_title is required.")

    query_lower = query.lower()
    exact_matches = [title for title in titles if title.lower() == query_lower]
    if exact_matches:
        return exact_matches[0]

    contains_matches = [title for title in titles if query_lower in title.lower()]
    if contains_matches:
        return contains_matches[0]

    from difflib import get_close_matches

    matches = get_close_matches(query, titles, n=1, cutoff=0.35)
    if not matches:
        raise HTTPException(status_code=404, detail=f"No close movie match found for '{query}'.")
    return matches[0]


def content_recommendations(title, n, artifacts):
    movies = artifacts["movies"]
    movie_genres_matrix = artifacts["movie_genres_matrix"]

    movie_row = movies.index[movies["title"] == title]
    if len(movie_row) == 0:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' was not found.")

    idx = int(movie_row[0])
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

    if user_id not in user_id_to_idx:
        raise HTTPException(status_code=404, detail=f"User ID {user_id} was not found.")

    user_idx = user_id_to_idx[user_id]
    scores = user_matrix[user_idx].dot(movie_matrix.T)
    rated_movie_indices = ratings_matrix[user_idx].nonzero()[1]
    scores[rated_movie_indices] = -np.inf
    top_movie_indices = np.argsort(scores)[-n:][::-1]
    return [movie_id_to_title[movie_idx_to_id[int(i)]] for i in top_movie_indices]


@app.get("/")
def home():
    return {
        "message": "Movie Recommender API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "artifacts_available": ARTIFACT_PATH.exists()}


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    artifacts = load_artifacts()
    titles = artifacts["titles"]
    matched_title = None

    if request.strategy == "matrix_factorization":
        if request.user_id is None:
            raise HTTPException(
                status_code=400,
                detail="user_id is required for matrix_factorization strategy.",
            )
        recommendations = matrix_factorization_recommendations(
            request.user_id,
            request.n_recommendations,
            artifacts,
        )
    else:
        matched_title = find_title(request.movie_title, titles)
        if request.strategy == "content":
            recommendations = content_recommendations(
                matched_title,
                request.n_recommendations,
                artifacts,
            )
        elif request.strategy == "collaborative":
            recommendations = collaborative_recommendations(
                matched_title,
                request.n_recommendations,
                artifacts,
            )
        elif request.strategy == "auto" and request.user_id in artifacts["user_id_to_idx"]:
            recommendations = matrix_factorization_recommendations(
                request.user_id,
                request.n_recommendations,
                artifacts,
            )
        else:
            recommendations = content_recommendations(
                matched_title,
                request.n_recommendations,
                artifacts,
            )

    return {
        "strategy": request.strategy,
        "matched_title": matched_title,
        "user_id": request.user_id,
        "recommendations": recommendations,
    }
