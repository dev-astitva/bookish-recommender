# Bookish Recommender

A Flask web app that recommends books using collaborative filtering, built on the
[Book-Crossing dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).
It has two pages: a **Top Rated** home page and a **Recommend** page where you type a
book you liked and get back similar titles other readers with similar taste also enjoyed.

This README walks through *how* the recommendation model works, in plain language —
useful if you're learning recommender systems, not just running the app.

---

## What's in this repo

```
.
├── app.py                     # Flask app (serves the two pages)
├── book-recomm-sys.ipynb      # Notebook: data cleaning + model building + export
├── books.pkl                  # Book metadata (title, author, cover)
├── popular.pkl                # Precomputed top-50 popular books
├── pt.pkl                     # Pivot table (books x users) used for similarity
├── similarity_scores.pkl      # Precomputed book-to-book similarity matrix
├── templates/
│   ├── index.html
│   └── recommend.html
├── images/                    # Fallback cover image, served at /images/<file>
└── requirements.txt
```

The notebook does all the heavy lifting once, offline, and saves its results as
`.pkl` (pickle) files. The Flask app just loads those files and looks things up —
that's why recommendations appear instantly instead of being recomputed on every
request.

---

## The dataset

Three CSVs, joined together:

- **Books** — ISBN, title, author, year, publisher, cover image URLs.
- **Users** — user ID, location, age.
- **Ratings** — which user rated which book (by ISBN), and what rating (0–10) they gave.

Ratings is the important one — it's the raw signal the whole model is built from.

---

## Part 1: The "Top Rated" page (popularity-based recommending)

This is the simplest kind of recommender: just show what's popular. But "popular"
needs two numbers, not one:

1. **How many people rated it** (`num_ratings`) — a book only two people rated
   isn't reliably "good," it's just unproven.
2. **How highly they rated it on average** (`avg_ratings`).

The notebook groups all ratings by book title to get both numbers, then keeps only
books with **at least 50 ratings** before sorting by average rating. That 50-rating
floor matters — without it, a book with a single 10/10 rating would outrank a book
with 2,000 ratings averaging 9.2, which isn't a fair comparison. The top 50 books
that clear the bar become the home page.

This approach has no idea what a book is *about*, and it recommends the same list
to everyone. That's what the second page fixes.

---

## Part 2: The "Recommend" page (collaborative filtering)

The idea behind collaborative filtering: **if two books tend to be rated similarly
by the same people, they're probably similar books** — regardless of genre, author,
or any text description. It's entirely pattern-based, learned from user behavior.

Here's the pipeline, step by step:

### 1. Filter down to trustworthy signal

Raw rating data is noisy — most users only rate a book or two, and most books only
get a handful of ratings. Both make similarity comparisons unreliable, so the
notebook filters twice:

- **Trusted users**: only keep users who've rated more than 200 books. Someone who's
  rated hundreds of books gives a much more reliable signal about "taste" than
  someone who rated one.
- **Famous books**: among those trusted users' ratings, only keep books with 50+
  ratings. A book two people rated can't be meaningfully compared to anything.

### 2. Build the ratings matrix (pivot table)

The filtered ratings get reshaped into a table: **one row per book, one column per
user, each cell is that user's rating for that book**. This is the classic
"user-item matrix" every collaborative filtering system is built on. Most cells are
empty (a given user has only rated a tiny fraction of all books), so it's filled
with 0 to keep the math simple — this is called a **sparse matrix**.

### 3. Mean-center each user's ratings

Different people use rating scales differently — some rate everything 7+, others
are harsher and rarely go above 5. If we compared raw ratings directly, we'd mostly
be measuring "how generous is this rater," not "how similar are these two books."

To fix this, each user's ratings are centered around **their own average rating**
(subtracting their mean from each of their ratings). Now a rating means "better or
worse than this person's usual," which is comparable across users.

### 4. Down-weight users who rate almost everything (inverse user frequency)

Some users rate huge swaths of the catalog indiscriminately. Their ratings carry
less information about what makes two *specific* books similar, since they'd rate
almost anything. The notebook applies a weighting scheme borrowed from the "inverse
document frequency" idea used in text search: users who've rated more of the
catalog get down-weighted, so their input counts for less, and choosier raters —
whose ratings are more informative — count for more.

### 5. Reduce to latent factors (SVD)

The matrix is still huge and mostly empty, which makes direct comparisons noisy —
two books might look dissimilar just because they happen to share few common
raters, not because they're actually different. **Truncated SVD** (a matrix
factorization technique, related to how classic recommender systems like the
Netflix Prize solutions worked) compresses each book down to 50 numbers — its
"latent factors." These don't map to anything human-readable like "genre" or
"length," but they capture the underlying patterns in how books get rated
together, in a much denser, less noisy form.

### 6. Compare books with cosine similarity

With every book now represented as a compact vector of latent factors, **cosine
similarity** measures how closely two books' vectors point in the same direction.
The closer to 1, the more similar the rating pattern between two books. This gives
a full book-by-book similarity matrix, computed once and saved as
`similarity_scores.pkl`.

### 7. Recommend

Given a book title, the app looks up its row in the similarity matrix, sorts every
other book by similarity score, and returns the top matches (excluding the book
itself, which is always most similar to itself).

---

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000
