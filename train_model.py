import os
from os.path import exists, join
import numpy as np
import gzip
import requests
from pathlib import Path
from random import shuffle
from time import time
import matplotlib.pyplot as plt
import sys

from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import ParameterGrid

# Hyperparameters LDA
n_samples = 2000
n_features = 5000
n_components = 15
batch_size = 128
max_df = 0.7
min_df = 20
init = "nndsvda"

# Hyperparameters plotting
n_top_words = 20


def download_ft_vectors():
    path = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
    processed_dir = join(os.getcwd(), 'data')
    Path(processed_dir).mkdir(exist_ok=True, parents=True)
    gz_file = join(processed_dir, path.split('/')[-1])
    if not exists(gz_file):
        with open(gz_file, "wb") as f:
            f.write(requests.get(path).content)
    print("Word vectors available!")
    return gz_file


def get_feature_vectors(gz_file, feature_names):
    m = []
    sorted_feature_names = []
    first_line = True
    c = 0
    with gzip.open(gz_file, 'rt') as fin:
        for l in fin:
            if first_line:
                first_line = False
                continue
            fs = l.rstrip('\n').split()
            word = fs[0]
            if word in feature_names:
                vec = np.array([float(v) for v in fs[1:]])
                sorted_feature_names.append(word)
                m.append(vec)
            c += 1
            if c > 50000:  # Only consider top 50k words for efficiency
                break
    return sorted_feature_names, np.array(m)


def compute_coherence(top_words, sorted_feature_list, m):
    feats_idx = [sorted_feature_list.index(w) for w in top_words if w in sorted_feature_list]
    truncated_m = m[feats_idx, :]
    cosines = 1 - pairwise_distances(truncated_m,
                                     metric="cosine") / 2  # Dividing by 2 to avoid negative values. See https://stackoverflow.com/questions/37454785/how-to-handle-negative-values-of-cosine-similarities
    return np.mean(cosines)


def return_coherence_list(model, feature_names, n_top_words, sorted_feature_list, m):
    coherences = []
    for topic_idx, topic in enumerate(model.components_):
        print("---> Computing coherence for topic", topic_idx)
        top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
        top_features = [feature_names[i] for i in top_features_ind]
        coherence = compute_coherence(top_features, sorted_feature_list, m)
        print(top_features, coherence)
        coherences.append(coherence)
    return coherences


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(4, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.3)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 14})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=8)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=14)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def load_data(doc_length=200):
    f = open(sys.argv[1], "r", encoding="utf-8-sig")
    # f = open("./source/input/combined/tokens_verb_noun.txt", "r", encoding="utf-8-sig")
    # f = open("./source/input/combined/tokens_all.txt", "r", encoding="utf-8-sig")
    docs_original = f.read()
    docs_lower = docs_original.lower().split()
    docs = [' '.join(docs_lower[start:start + doc_length]) for start in range(0, len(docs_lower), doc_length)]
    shuffle(docs)
    return docs


def get_tfs(data=None, nfeats=None, max_df=None, min_df=None):
    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(
        max_df=max_df, min_df=min_df, max_features=nfeats, stop_words="english"
    )
    tfs = tf_vectorizer.fit_transform(data)
    return tf_vectorizer, tfs


def run_LDA(data=None, max_df=None, min_df=None, nfeats=None, n_components=None):
    tf_vectorizer, tfs = get_tfs(data=data, nfeats=nfeats, max_df=max_df, min_df=min_df)
    print("\nFitting LDA models with tf features, n_samples=%d and n_features=%d..." % (n_samples, nfeats))
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=5,
        learning_method="online",
        learning_offset=10.0,
        random_state=0,
    )
    t0 = time()
    lda.fit(tfs)
    print("done in %0.3fs." % (time() - t0))
    perplexity = lda.perplexity(tfs)
    # print("Perplexity:",perplexity)
    return tf_vectorizer, lda, perplexity


print("Downloading word vectors...")
gz_path = download_ft_vectors()

# param_grid = {'doc_length': [50, 100, 200, 300, 400, 500], 'max_df': [0.5, 0.6, 0.7, 0.8], 'min_df': [0.01, 0.05, 0.1, 0.2, 0.3],
#               'n_features': [1000, 2000, 3000, 5000, 7000, 10000], 'n_components': [8, 12, 16, 24, 32, 48, 64]}
param_grid = {'doc_length': [50,100], 'max_df': [0.7], 'min_df': [0.2],
              'n_features': [1000], 'n_components': [12,24,36,48,96,192,256]}
# param_grid = {'doc_length': [50,100], 'max_df': [0.5,0.6,0.7,0.8], 'min_df': [0.1,0.2,0.3], 'n_features': [1000,2000,3000,5000], 'n_components':[24,48,96]}

grid = ParameterGrid(param_grid)

perplexities = []
coherences = []

for p in grid:
    print("\n", p)
    print("Loading dataset...")
    data = load_data(doc_length=p['doc_length'])
    data_samples = data[:n_samples]
    tf_vectorizer, lda, perplexity = run_LDA(data=data_samples, max_df=p['max_df'], min_df=p['min_df'],
                                             nfeats=p['n_features'], n_components=p['n_components'])
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    sorted_feature_names, m = get_feature_vectors(gz_path, tf_feature_names)
    print("Perplexity", perplexity)
    perplexities.append(perplexity)
    coherence = np.mean(return_coherence_list(lda, tf_feature_names, n_top_words, sorted_feature_names, m))
    print("Coherence", coherence)
    coherences.append(coherence)

# best = np.argmax(coherences)

# Sort coherences descending, and perplexities ascending, in order to find at which index do we have a combo of
# the highest coherence and lowest perplexity (to find a potentially most balanced model)
# coherences_sorted_desc = sorted(range(len(coherences)), key=coherences.__getitem__, reverse=True)
# perplexities_sorted_asc = sorted(range(len(perplexities)), key=perplexities.__getitem__)
#
# best_coh_perp_combo = 0
# for i, coh_index in enumerate(coherences_sorted_desc):
#     if perplexities_sorted_asc[i] == coh_index:
#         best_index = coh_index
#         break

best_results = {"coherence": np.argmax(coherences), "perplexity": np.argmin(perplexities)}

for result_name, best in best_results.items():
    p = grid[best]

    print(sys.argv[1].split("/")[-1], f"\n\n[{result_name.upper()}] BEST HYPERPARAMETERS:", grid[best], "COHERENCE:",
          coherences[best], "PERPLEXITY:", perplexities[best])

    # tf_vectorizer, lda, perplexity = run_LDA(data=data_samples, max_df=0.7, min_df=50, nfeats=5000, n_components=8)
    tf_vectorizer, lda, perplexity = run_LDA(data=data_samples, max_df=p['max_df'], min_df=p['min_df'],
                                             nfeats=p['n_features'], n_components=p['n_components'])
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    sorted_feature_names, m = get_feature_vectors(gz_path, tf_feature_names)
    cohs = return_coherence_list(lda, tf_feature_names, n_top_words, sorted_feature_names, m)
    # plot_top_words(lda, tf_feature_names, n_top_words, "Topics in LDA model")
