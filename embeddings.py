import numpy as np


def load_word_emb(word_index, embedding_file):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(
        get_coefs(*o.rstrip().split(" ")) for o in open(embedding_file, encoding="utf8") if
        o.rstrip().split(" ")[0] in word_index)
    return embeddings_index


def get_emb_matrix(word_index, max_features, embedding_file):
    embeddings_index = load_word_emb(word_index, embedding_file)
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        if embedding_vector is None: print(word)

    return embedding_matrix
