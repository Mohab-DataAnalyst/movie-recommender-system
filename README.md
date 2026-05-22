# Movie Recommender System

This project builds a movie recommendation system using ratings and genre data. It includes exploratory analysis, collaborative filtering, content-based recommendations, matrix factorization, and simple deployment demos using FastAPI and Streamlit.

## Project Overview

The goal of this project is to recommend movies based on either a selected movie title or an existing user ID. The notebook explores the MovieLens-style dataset and experiments with multiple recommendation strategies. The deployment files package the recommender into a simple API and an interactive web app.

## Dataset

The project uses these files:

- `movies.csv`: movie titles and genres
- `ratings.csv`: user ratings for movies
- `tags.csv`: user-generated movie tags
- `links.csv`: external movie IDs

## Recommendation Strategies

- Content-based filtering: recommends movies with similar genres.
- Collaborative filtering: recommends movies that have similar user-rating patterns.
- Matrix factorization: uses TruncatedSVD to recommend movies to a known user.
- Auto mode: uses matrix factorization when a valid user ID is provided, otherwise falls back to content-based recommendations.

## Workflow

1. Loaded the movies and ratings datasets.
2. Explored movie, user, rating, and genre distributions.
3. Created a sparse user-movie ratings matrix.
4. Built movie-to-movie collaborative filtering using nearest neighbors.
5. Built content-based filtering using one-hot encoded genres and cosine similarity.
6. Built matrix factorization recommendations using TruncatedSVD.
7. Saved reusable recommendation artifacts as `recommender_artifacts.pkl`.
8. Added FastAPI and Streamlit apps for simple deployment demos.

## Tech Stack

- Python
- pandas
- NumPy
- SciPy
- scikit-learn
- joblib
- FastAPI
- Streamlit
- Jupyter Notebook

## Project Structure

```text
.
+-- Recommender.ipynb
+-- movies.csv
+-- ratings.csv
+-- tags.csv
+-- links.csv
+-- train_model.py
+-- recommender_artifacts.pkl
+-- app.py
+-- streamlit_app.py
+-- requirements.txt
+-- README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Build and save the recommender artifacts:

```bash
python train_model.py
```

Run the FastAPI app:

```bash
uvicorn app:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## API Example

Send a `POST` request to `/recommend`:

```json
{
  "strategy": "auto",
  "movie_title": "Toy Story",
  "user_id": 25,
  "n_recommendations": 5
}
```

Example response:

```json
{
  "strategy": "auto",
  "matched_title": "Toy Story (1995)",
  "user_id": 25,
  "recommendations": [
    "Forrest Gump (1994)",
    "Pulp Fiction (1994)",
    "Silence of the Lambs, The (1991)"
  ]
}
```

## What I Learned

This project helped me practice recommender system concepts, sparse matrices, cosine similarity, nearest neighbors, matrix factorization, and turning notebook logic into a simple deployable demo.
