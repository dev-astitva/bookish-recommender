# Bookish — Book Recommender System

A Flask web app that recommends books using collaborative filtering, built
on the Book-Crossing dataset. Includes a "Top Rated" home page and a
"Recommend" page that suggests similar titles based on user rating patterns.

## Features

- **Home page** — top-rated books (min. 50 ratings), sorted by average score.
- **Recommend page** — type in a title, get similar books via item-based
  collaborative filtering (cosine similarity on a user–book pivot table).
- Data pipeline and model built in `book-recomm-sys.ipynb`.

## Project structure

```
.
├── app.py                     # Flask app
├── book-recomm-sys.ipynb      # Notebook: data prep + model + pickle export
├── books.pkl                  # Book metadata
├── popular.pkl                # Precomputed top-50 popular books
├── pt.pkl                     # Pivot table (books x users) used for similarity
├── similarity_scores.pkl      # Precomputed cosine similarity matrix
├── templates/
│   ├── index.html
│   └── recommend.html
├── images/                    # Fallback cover image(s) served at /images/<file>
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000

## Notes

- `app.py` hot-reloads the `.pkl` files when their contents change on disk,
  so re-running the notebook and re-exporting doesn't require restarting
  the server.
- Add a fallback cover (`image1.jpg`) to the `images/` folder — it's used
  whenever a book's cover URL fails to load.
