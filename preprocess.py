import io

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
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300  # The dimension of the word embeddings
BATCH_SIZE = 200  # The batch size to be chosen for training the model.
LSTM_DIM = 300  # The dimension of the representations learnt by the LSTM model
DROPOUT = 0.2  # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = 10  # Number of epochs to train a model for

label2index = {}
index2label = {}


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


def compute_features(articles, ref_articles_id, span_starts, span_ends):
    # only one feature, the length of the span
    # data =
    print(type(span_starts), len(span_starts))
    print(type(span_ends), len(span_ends))
    data = []
    for i, ref_id in enumerate(ref_articles_id):
        # print(articles[ref_id], span_starts[i], span_ends[i])
        article = articles[ref_id]
        article_span = article[int(span_starts[i]):int(span_ends[i])]
        data.append([article_span, ])
    return


def preprocess():
    pass


### MAIN ###

# loading articles' content from *.txt files in the train folder
articles = read_articles_from_file_list(train_folder)
# print(articles['111111111'])
# exit()

# loading gold labels, articles ids and sentence ids from files *.task-TC.labels in the train labels folder
ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels = read_predictions_from_file(train_labels_file)
print("Loaded %d annotations from %d articles" % (len(ref_span_starts), len(set(ref_articles_id))))

# compute one feature for each fragment, i.e. the length of the fragment, and train the model
train = compute_features(articles, ref_articles_id, ref_span_starts, ref_span_ends)
