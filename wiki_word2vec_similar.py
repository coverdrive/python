import pandas as pd
import numpy as np

subset_words = -1
word = "assuage"
print "Given word is %s" % word
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
words = list(df.index)
word_index = words.index(word)
vecs = df.as_matrix()
vec_norms = np.linalg.norm(vecs, axis=1)

word_vec = vecs[word_index]
word_vec_norm = np.linalg.norm(word_vec)
# cosines = vecs.dot(vecs.T) / np.outer(vec_norms, vec_norms)
cosines = vecs.dot(word_vec) / vec_norms / word_vec_norm
nearest_words = [words[i] for i in
                 cosines.argsort()[-num_nearest_words:]][::-1]
print "%d nearest-by-context words are:" % num_nearest_words
print nearest_words
