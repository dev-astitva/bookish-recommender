from flask import Flask, render_template, request, send_from_directory
import pickle
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'images')

# Must match the slice used in book-recomm-sys.ipynb's recommend() cell
# ([1:6] => 5 results). Keeping this as a single constant means the app
# and the notebook can't quietly drift apart on how many results to show.
TOP_N = 8

DATA_FILES = {
    'popular_df': 'popular.pkl',
    'pt': 'pt.pkl',
    'books': 'books.pkl',
    'similarity_scores': 'similarity_scores.pkl',
}

_data = {}
_mtimes = {}


def _load_data(force=False):
    for key, filename in DATA_FILES.items():
        path = os.path.join(BASE_DIR, filename)
        mtime = os.path.getmtime(path)
        if force or key not in _data or _mtimes.get(key) != mtime:
            with open(path, 'rb') as f:
                _data[key] = pickle.load(f)
            _mtimes[key] = mtime
    return _data


# Warm the cache at startup so the first request isn't slower than the rest.
_load_data(force=True)

app = Flask(__name__)


@app.route('/images/<path:filename>')
def serve_image(filename):
    # serves files straight from the local images/ folder next to app.py
    return send_from_directory(IMAGES_DIR, filename)


@app.route('/')
def index():
    data = _load_data()
    popular_df = data['popular_df']
    return render_template(
        'index.html',
        book_name=list(popular_df['title'].values),
        author=list(popular_df['author'].values),
        image=list(popular_df['Image-URL-M'].values),
        votes=list(popular_df['num_ratings'].values),
        rating=list(popular_df['avg_ratings'].values)
    )


@app.route('/recommend')
def recommend_ui():
    data = _load_data()
    pt = data['pt']
    return render_template('recommend.html', titles=list(pt.index))


@app.route('/recommend_books', methods=['post'])
def recommend():
    data = _load_data()
    pt = data['pt']
    books = data['books']
    similarity_scores = data['similarity_scores']

    user_input = request.form.get('user_input', '').strip()

    matches = np.where(pt.index == user_input)[0]

    if len(matches) == 0:
        return render_template(
            'recommend.html',
            error=f'"{user_input}" isn\'t in our catalog yet. Double check the spelling or try another title.',
            book_name=user_input,
            titles=list(pt.index)
        )

    index = matches[0]
    similar_items = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:TOP_N + 1]

    results = []
    for i in similar_items:
        item = []
        temp_df = books[books['title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('title')['title'].values))
        item.extend(list(temp_df.drop_duplicates('title')['author'].values))
        item.extend(list(temp_df.drop_duplicates('title')['Image-URL-M'].values))
        results.append(item)

    return render_template('recommend.html', data=results, book_name=user_input, titles=list(pt.index))


if __name__ == "__main__":
    app.run(debug=True)
