from load_embeddings import load_word_vectors

glove_path = '/home/lena/diplomatiki/06_datasets/06_glove/glove.42B.300d.txt'
word2idx, dataset_data = load_word_vectors(glove_path, take= 10000)
# print(word2idx, dataset_data)