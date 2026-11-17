import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import run


def test_clean_lowercases_and_removes_urls_but_keeps_accents():
    assert run.clean("Superbe FILM, vu sur http://exemple.fr/abc !") == "superbe film, vu sur   !"
    assert "é" in run.clean("Émouvant")


def test_clean_keeps_negation_words():
    # "pas" and "ne" matter for sentiment, so they must not be removed
    out = run.clean("Ce n'est PAS un bon film")
    assert "pas" in out


def test_word_bigrams_can_capture_a_simple_negation():
    texts = ["pas mal du tout", "pas mal ce film", "très mal joué", "mal écrit et mal joué",
             "excellent film", "superbe film", "film ennuyeux", "vraiment nul"]
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(texts)
    model = LinearSVC(C=1.0).fit(X, y)
    assert model.predict(vec.transform(["pas mal"]))[0] == 1
    assert model.predict(vec.transform(["très mal"]))[0] == 0
