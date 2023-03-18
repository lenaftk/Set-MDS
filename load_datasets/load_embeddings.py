import errno
import os
import numpy


def load_word_vectors(file, take=-1):
    """
    Read the word vectors from a text file
    Args:
        file (): the filename
    Returns:
        word2idx (dict): dictionary of words to ids
        embeddings (numpy.ndarray): the word embeddings matrix
    """
    # create the necessary dictionaries and the word embeddings matrix
    if os.path.exists(file):
        print('Indexing file {} ...'.format(file))

        word2idx = {}  # dictionary of words to ids
        embeddings = []  # the word embeddings matrix

        # read file, line by line
        # with open(file, "r", encoding="utf-8") as f:
        with open(file, "r") as f:
            for i, line in enumerate(f, 0):
                values = line.split(" ")
                word = values[0]
                vector = numpy.asarray(values[1:], dtype='float64')
                word2idx[word] = i
                embeddings.append(vector)
                if 0 < take <= i + 1:
                    break

            print(set([len(x) for x in embeddings]))

            print('Found %s word vectors.' % len(embeddings))
            embeddings = numpy.array(embeddings, dtype='float64')

        return word2idx, embeddings
    else:
        print("{} not found!".format(file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), file)
