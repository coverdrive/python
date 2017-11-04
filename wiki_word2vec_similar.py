import pandas as pd
import numpy as np


def get_vec_from_word(df, word):
    return df.loc[word]


def get_nearest_words(df, vec, num_nearest_words):
    vecs = df.as_matrix()
    vec_norms = np.linalg.norm(vecs, axis=1)
    vec_norm = np.linalg.norm(vec)
    cosines = vecs.dot(vec) / vec_norms / vec_norm
    nearest_word_indices = cosines.argsort()[-num_nearest_words:][::-1]
    return df.index[nearest_word_indices]

if __name__ == "__main__":
    subset_words = -1
    num_nearest_words = 20

    filename = "/Users/z001xdc/Downloads/wiki_w2v.vec"
    df = pd.io.parsers.read_csv(
        filename,
        sep=' ',
        engine='c',
        skiprows=1,
        index_col=0,
        header=None,
        nrows=(subset_words if subset_words > 0 else None)
    )

    # word = "assuage"
    # print "Given word is %s" % word
    # x_vec = get_vec_from_word(df, word)

    # ireland_vec = get_vec_from_word(df, "ireland")
    # dublin_vec = get_vec_from_word(df, "dublin")
    # moscow_vec = get_vec_from_word(df, "moscow")
    # x_vec = ireland_vec - dublin_vec + moscow_vec

    country = "india"
    entity = "language"
    country_vec = get_vec_from_word(df, country)
    entity_vec = get_vec_from_word(df, entity)
    x_vec = country_vec + entity_vec

    nearest_words = get_nearest_words(df, x_vec, num_nearest_words)
    print "%d nearest-by-context words are:" % num_nearest_words
    print nearest_words
