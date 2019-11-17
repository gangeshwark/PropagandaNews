import io
import json
import pickle
import socket
from pprint import pprint

import pandas as pd
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tqdm import tqdm

train_folder = "datasets/train-articles"  # check that the path to the datasets folder is correct,
dev_folder = "datasets/dev-articles"  # if not adjust these variables accordingly
train_labels_file = "datasets/train-task2-TC.labels"
dev_template_labels_file = "datasets/dev-task-TC-template.out"
task_TC_output_file = "baseline-output-TC.txt"

#
# Baseline for Task TC
#
# Our baseline uses a logistic regression classifier on one feature only: the length of the sentence.
#
# Requirements: sklearn, numpy
#

import codecs
import glob
import os.path

import numpy as np

gloveDir = ""
MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 140
EMBEDDING_DIM = 300  # The dimension of the word embeddings
BATCH_SIZE = 200  # The batch size to be chosen for training the model.
LSTM_DIM = 300  # The dimension of the representations learnt by the LSTM model
DROPOUT = 0.2  # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = 10  # Number of epochs to train a model for

label2index = {
    'Appeal_to_Authority': 0,
    'Appeal_to_fear-prejudice': 1,
    'Bandwagon,Reductio_ad_hitlerum': 2,
    'Black-and-White_Fallacy': 3,
    'Causal_Oversimplification': 4,
    'Doubt': 5,
    'Exaggeration,Minimisation': 6,
    'Flag-Waving': 7,
    'Loaded_Language': 8,
    'Name_Calling,Labeling': 9,
    'Repetition': 10,
    'Slogans': 11,
    'Thought-terminating_Cliches': 12,
    'Whataboutism,Straw_Men,Red_Herring': 13
}

index2label = {
    0: 'Appeal_to_Authority',
    1: 'Appeal_to_fear-prejudice',
    2: 'Bandwagon,Reductio_ad_hitlerum',
    3: 'Black-and-White_Fallacy',
    4: 'Causal_Oversimplification',
    5: 'Doubt',
    6: 'Exaggeration,Minimisation',
    7: 'Flag-Waving',
    8: 'Loaded_Language',
    9: 'Name_Calling,Labeling',
    10: 'Repetition',
    11: 'Slogans',
    12: 'Thought-terminating_Cliches',
    13: 'Whataboutism,Straw_Men,Red_Herring'
}


def getEmbeddingMatrix(wordIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}

    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(gloveDir, 'glove.840B.300d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embeddingVector = np.array([float(val) for val in values[1:]])
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    no_emb = []
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
        else:
            embeddingMatrix[i] = np.random.uniform(0, 1, 300)
            no_emb.append(word)

    no_emb = list(set(no_emb))
    with io.open(os.path.join(gloveDir, 'no_emb.txt'), encoding="utf8", mode='w') as output:
        output.writelines('\n'.join(no_emb))

    return embeddingMatrix


def read_articles_from_file_list(folder_name, file_pattern="*.txt"):
    """
    Read articles from files matching patterns <file_pattern> from
    the directory <folder_name>.
    The content of the article is saved in the dictionary whose key
    is the id of the article (extracted from the file name).
    Each element of <sentence_list> is one line of the article.
    """
    file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles = {}
    article_id_list, sentence_id_list, sentence_list = ([], [], [])
    for filename in sorted(file_list):
        article_id = os.path.basename(filename).split(".")[0][7:]
        with codecs.open(filename, "r", encoding="utf8") as f:
            articles[article_id] = f.read()
    return articles


def read_predictions_from_file(filename):
    """
    Reader for the gold file and the template output file.
    Return values are four arrays with article ids, labels
    (or ? in the case of a template file), begin of a fragment,
    end of a fragment.
    """
    articles_id, span_starts, span_ends, gold_labels = ([], [], [], [])
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, gold_label, span_start, span_end = row.rstrip().split("\t")
            articles_id.append(article_id)
            gold_labels.append(gold_label)
            span_starts.append(span_start)
            span_ends.append(span_end)
    return articles_id, span_starts, span_ends, gold_labels


def clean_text(text):
    # text = text.replace('\'', '')
    text = text.replace('‘', ' \' ')
    text = text.replace('’', ' \' ')
    text = text.replace('“', ' \" ')
    text = text.replace('”', ' \" ')

    text = text.replace('"', ' " ')
    text = text.replace('\'', ' \' ')

    text = text.replace('—', ' - ')
    text = text.replace('–', ' - ')
    text = text.replace('…', '...')
    text = text.strip()
    return text


def get_CrystalFeel_features(text):
    PORT_NUMBER = 8247
    IP_ADDRESS = '10.217.163.99'

    text = bytes(text, 'utf-8')
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP_ADDRESS, PORT_NUMBER))

    s.send(text)

    result = ''
    out = s.recv(1)
    while (out != b'\n'):
        result = result + str(out, 'utf-8')
        out = s.recv(1)

    s.close()
    result = json.loads(result)
    # pprint(result)
    # print(result['intensity_scores'])
    # print(type(result))
    return result


def compute_features(articles, ref_articles_id, span_starts, span_ends, use_emotion_features=False):
    # only one feature, the length of the span
    # data =
    new_data = pd.DataFrame(
        columns=['article_id', 'article_span', 'span_start', 'span_end', 'tAnger', 'tFear',
                 'tJoy', 'tSadness', 'tValence', 'tEmotion', 'tEmotionScore', 'tSentiment', 'tSentimentScore'])
    print(type(span_starts), len(span_starts))
    print(type(span_ends), len(span_ends))
    data = []
    article_spans = []
    for i, ref_id in tqdm(enumerate(ref_articles_id)):
        # print(articles[ref_id], span_starts[i], span_ends[i])
        article = articles[ref_id]
        article_span = clean_text(article[int(span_starts[i]):int(span_ends[i])])
        data.append([article_span])
        article_spans.append(article_span)
        if article_span == '':
            with open('error.txt', 'a+') as error_file:
                error_file.write("{0}\t{1}\t{2}\n".format(ref_id, article_span, 'EMPTY STRING'))
            if use_emotion_features:
                new_data.loc[i] = [ref_id, article_span, int(span_starts[i]), int(span_ends[i]),
                                   0.0, 0.0, 0.0, 0.0, 0.0, 'no specific emotion', 0, 'neutral', 0]
            continue
        if use_emotion_features:
            features = get_CrystalFeel_features(article_span)
            if features['status'] != 'success':
                with open('error.txt', 'a+') as error_file:
                    error_file.write("{0}\t{1}\t{2}\n".format(ref_id, article_span, features['status']))
                new_data.loc[i] = [ref_id, article_span, int(span_starts[i]), int(span_ends[i]),
                                   0.0, 0.0, 0.0, 0.0, 0.0, 'no specific emotion', 0, 'neutral', 0]
                continue

            # print(features)
            # print(i)
            intensity_scores = features['intensity_scores']
            labels = features['labels']
            new_data.loc[i] = [ref_id, article_span, int(span_starts[i]), int(span_ends[i]), intensity_scores['tAnger'],
                               intensity_scores['tFear'], intensity_scores['tJoy'],
                               intensity_scores['tSadness'], intensity_scores['tValence'], labels['tEmotion'],
                               labels['tEmotionScore'], labels['tSentiment'],
                               labels['tSentimentScore']]

    # print(article_spans)
    if use_emotion_features:
        return article_spans, new_data
    return article_spans, None


if __name__ == '__main__':
    use_emotion_features = False
    ### MAIN ###

    # loading articles' content from *.txt files in the train folder
    articles = read_articles_from_file_list(train_folder)
    dev_articles = read_articles_from_file_list(dev_folder)

    # computing the predictions on the development set
    # print(articles['111111111'])
    # exit()

    # loading gold labels, articles ids and sentence ids from files *.task-TC.labels in the train labels folder
    ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels = read_predictions_from_file(train_labels_file)
    dev_article_ids, dev_span_starts, dev_span_ends, dev_labels = read_predictions_from_file(dev_template_labels_file)
    print("Loaded %d annotations from %d articles" % (len(ref_span_starts), len(set(ref_articles_id))))

    # compute one feature for each fragment, i.e. the length of the fragment, and train the model
    articles, train_emo_feat = compute_features(articles, ref_articles_id, ref_span_starts, ref_span_ends,
                                                use_emotion_features)
    dev_articles, dev_emo_feat = compute_features(dev_articles, dev_article_ids, dev_span_starts, dev_span_ends,
                                                  use_emotion_features)
    # exit()

    if use_emotion_features:
        print(train_emo_feat.shape)
        train_emo_feat.to_csv('datasets/train_articles_emotion_features.csv', index_label='index',
                              columns=['article_id', 'article_span', 'span_start', 'span_end', 'tAnger', 'tFear',
                                       'tJoy', 'tSadness', 'tValence', 'tEmotion', 'tEmotionScore', 'tSentiment',
                                       'tSentimentScore', 'label'])
        dev_emo_feat.to_csv('datasets/dev_articles_emotion_features.csv', index_label='index',
                            columns=['article_id', 'article_span', 'span_start', 'span_end', 'tAnger', 'tFear',
                                     'tJoy', 'tSadness', 'tValence', 'tEmotion', 'tEmotionScore', 'tSentiment',
                                     'tSentimentScore'])
        try:
            train_emo_feat['label'] = train_gold_labels
            train_emo_feat.to_csv('datasets/train_articles_emotion_features.csv', index_label='index',
                                  columns=['article_id', 'article_span', 'span_start', 'span_end', 'tAnger', 'tFear',
                                           'tJoy', 'tSadness', 'tValence', 'tEmotion', 'tEmotionScore', 'tSentiment',
                                           'tSentimentScore', 'label'])
        except:
            print('ERRORRRRRR!!!!')

    if True:
        train_emo_feat = pd.read_csv('datasets/train_articles_emotion_features.csv', index_col='index')
        dev_emo_feat = pd.read_csv('datasets/dev_articles_emotion_features.csv', index_col='index')
        # pd.read_csv()

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(articles + dev_articles)
    articles_id = tokenizer.texts_to_sequences(articles)
    dev_articles_id = tokenizer.texts_to_sequences(dev_articles)
    # print(articles_id)
    # print(dev_articles_id)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")
    embeddingMatrix = getEmbeddingMatrix(wordIndex)

    train_seq_len = []
    dev_seq_len = []
    for x in articles_id:
        train_seq_len.append(len(x))

    for x in dev_articles_id:
        dev_seq_len.append(len(x))

    max_len = max(train_seq_len + dev_seq_len)
    print(max_len)

    articles_id = pad_sequences(articles_id, maxlen=MAX_SEQUENCE_LENGTH)
    dev_articles_id = pad_sequences(dev_articles_id, maxlen=MAX_SEQUENCE_LENGTH)
    pprint(set(train_gold_labels))

    labels = [label2index[x] for x in train_gold_labels]
    labels = to_categorical(np.asarray(labels))
    train_seq_len = np.array(train_seq_len)
    dev_seq_len = np.array(dev_seq_len)

    data_path = './processed_data/'

    # save train data
    pickle.dump(articles_id, open(data_path + 'train_x.p', 'wb'))
    pickle.dump(train_seq_len, open(data_path + 'train_seq_len.p', 'wb'))
    pickle.dump(labels, open(data_path + 'train_y.p', 'wb'))

    # save dev data
    pickle.dump(dev_articles_id, open(data_path + 'dev_x.p', 'wb'))
    pickle.dump(dev_seq_len, open(data_path + 'dev_seq_len.p', 'wb'))

    pickle.dump(embeddingMatrix, open(data_path + 'GloVe_Embeddings.p', 'wb'))

    if use_emotion_features or True:
        train = train_emo_feat[['tAnger', 'tFear', 'tJoy', 'tSadness',
                                'tValence', 'tEmotionScore', 'tSentimentScore']]
        dev = dev_emo_feat[['tAnger', 'tFear', 'tJoy', 'tSadness',
                            'tValence', 'tEmotionScore', 'tSentimentScore']]

        pickle.dump(train, open(data_path + 'train_x_emotion.p', 'wb'))
        pickle.dump(dev, open(data_path + 'dev_x_emotion.p', 'wb'))
