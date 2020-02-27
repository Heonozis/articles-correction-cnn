import numpy as np
import json
import pandas as pd


def embed_word(sentence, word_index, window_radius):
    embedding = []
    for i in range(1, window_radius + 1):
        if word_index - i < 0:
            embedding = [sentence[0]] + embedding
        else:
            embedding = [sentence[word_index-i]] + embedding
        if word_index + i > len(sentence) - 1:
            embedding = embedding + [sentence[len(sentence)-1]]
        else:
            embedding = embedding + [sentence[word_index + i]]

    return embedding


def embed_dataset(path, targets, window_radius, one_hot_labels=True):
    with open(path) as json_data:
        data = json.load(json_data)

    embeded_data = []
    labels = []
    for sentence in data:
        for word_index, word in enumerate(sentence):
            if word in targets:
                label = np.zeros(len(targets))
                label_index = targets.index(word)
                if one_hot_labels:
                    label[label_index] = 1
                    labels.append(label)
                else:
                    labels.append(label_index)
                embedding = embed_word(sentence, word_index, window_radius)
                embeded_data.append(embedding)
    return np.array(embeded_data), np.array(labels).astype(int)


def glove_encode_dataset(dataset, glove):
    glove_dataset = []
    for embedding in dataset:
        embedding_representation = []
        for word in embedding:
            word_representation = glove[word]
            embedding_representation = embedding_representation + word_representation
        glove_dataset.append(embedding_representation)
    return glove_dataset


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_ngrams_by_word(word, ngrams):
    if word in ngrams:
        return [
            [word, ngrams[word]]
        ]
    else:
        return []


def is_noun(word, singular, plural, uncountable):
    noun = {
        'singular': 0,
        'plural': 0,
        'uncountable': 0
    }

    if word in singular:
        noun['singular'] = 1

    if word in plural:
        noun['plural'] = 1

    if word in uncountable:
        noun['uncountable'] = 1

    return noun

from nltk import pos_tag

def encode_article_with_ngrams(word_index, sentence, ngrams, articles, pos_tags=None):
    vector = []
    for article in articles:
        if word_index - 2 > 0:
            left_trigram_search_word = sentence[word_index-2] + ' ' + sentence[word_index-1] + ' ' + article
            left_trigram = get_ngrams_by_word(left_trigram_search_word, ngrams)
            if len(left_trigram) > 0:
                vector.append(left_trigram[0][1])
            else:
                vector.append(0.0)
        else:
            vector.append(0.0)

        if word_index - 1 > 0 and word_index + 1 < len(sentence):
            center_trigram_search_word = sentence[word_index-1] + ' ' + article + ' ' + sentence[word_index+1]
            center_trigram = get_ngrams_by_word(center_trigram_search_word, ngrams)
            if len(center_trigram) > 0:
                vector.append(center_trigram[0][1])
            else:
                vector.append(0.0)
        else:
            vector.append(0.0)

        if word_index + 2 < len(sentence):
            right_trigram_search_word = article + ' ' + sentence[word_index+1] + ' ' + sentence[word_index+2]
            right_trigram = get_ngrams_by_word(right_trigram_search_word, ngrams)
            if len(right_trigram) > 0:
                vector.append(right_trigram[0][1])
            else:
                vector.append(0.0)
        else:
            vector.append(0.0)

        if word_index - 1 > 0:
            left_bigram_search_word = sentence[word_index-1] + ' ' + article
            left_bigram = get_ngrams_by_word(left_bigram_search_word, ngrams)
            if len(left_bigram) > 0:
                vector.append(left_bigram[0][1])
            else:
                vector.append(0.0)
        else:
            vector.append(0.0)

        if word_index + 1 < len(sentence):
            right_bigram_search_word = article + ' ' + sentence[word_index+1]
            right_bigram = get_ngrams_by_word(right_bigram_search_word, ngrams)
            if len(right_bigram) > 0:
                vector.append(right_bigram[0][1])
            else:
                vector.append(0.0)
        else:
            vector.append(0.0)

    if pos_tags:
        pos_sentence = pos_tag(sentence)
        pos_vectors = np.zeros((4, len(pos_tags)))

        if word_index - 2 > 0:
            pos = pos_sentence[word_index-2][1]
            if pos in pos_tags:
                pos_index = pos_tags.index(pos)
                pos_vectors[0][pos_index] = 1

        if word_index - 1 > 0:
            pos = pos_sentence[word_index-1][1]
            if pos in pos_tags:
                pos_index = pos_tags.index(pos)
                pos_vectors[1][pos_index] = 1

        if word_index + 1 < len(sentence):
            pos = pos_sentence[word_index+1][1]
            if pos in pos_tags:
                pos_index = pos_tags.index(pos)
                pos_vectors[2][pos_index] = 1

        if word_index + 2 < len(sentence):
            pos = pos_sentence[word_index+2][1]
            if pos in pos_tags:
                pos_index = pos_tags.index(pos)
                pos_vectors[3][pos_index] = 1

        pos_vectors = list(pos_vectors.flat)

        vector = vector + pos_vectors

    return vector


def encode_sentences_with_ngrams(sentences, ngrams, articles, pos_tags=None):
    labels = []
    vectors = []

    for sentence in sentences:
        for word_index, word in enumerate(sentence):
            if word in articles:
                article_index = articles.index(word)
                labels.append(article_index)
                article_vector = encode_article_with_ngrams(word_index, sentence, ngrams, articles, pos_tags=pos_tags)
                vectors.append(article_vector)
    df = pd.DataFrame(vectors)

    return df, labels
