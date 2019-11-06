# Please use python 3.5 or above
import io
import os
import pickle
import re

import emoji
import gensim
from keras.layers.core import *
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

data_path = './data_new/'

# Path to training and testing data file. This data can be downloaded from a link, details of which will be provided.
trainDataPath = data_path + "train.txt"
# testDataPath = data_path + "devwithoutlabels.txt"
testDataPath = data_path + "testwithoutlabels.txt"
# Output file that will be generated. This file can be directly submitted.
solutionPath = data_path + "test.txt"
# Path to directory where GloVe file is saved.
gloveDir = data_path + ""

NUM_FOLDS = 2  # Value of K in K-fold Cross Validation
NUM_CLASSES = 4  # Number of classes - Happy, Sad, Angry, Others
MAX_NB_WORDS = 35000  # To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer
MAX_SEQUENCE_LENGTH = 100  # All sentences having lesser number of words than this will be padded
EMBEDDING_DIM = 300  # The dimension of the word embeddings
BATCH_SIZE = 200  # The batch size to be chosen for training the model.
LSTM_DIM = 300  # The dimension of the representations learnt by the LSTM model
DROPOUT = 0.2  # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = 10  # Number of epochs to train a model for

label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}


# TODO:
# 1. remove stop words.
def preprocessText(sent):
    """
    1. Check if a character is emoji
    :param sent:
    :return:
    """
    # 1. replace :), :(, :/, etc  with it's emoji version
    replace_emoji = {
        ':)': u'\U0001f642',  # happy
        '(:': u'\U0001f642',  # happy
        '=)': u'\U0001f642',  # happy
        ':]': u'\U0001f642',  # happy
        ':-)': u'\U0001f642',  # happy
        ':(': u'\U00002639',  # sad
        ':[': u'\U00002639',  # sad
        ':-[': u'\U00002639',  # sad
        ':-(': u'\U00002639',  # sad
        ':-c': u'\U00002639',  # sad
        ':/': u'\U0001f610',  # speechless
        ':\\': u'\U0001f610',  # speechless
        ':-/': u'\U0001f610',  # speechless
        ':-\\': u'\U0001f610',  # speechless
        ':|': u'\U0001f610',  # speechless
        ':@': u'\U0001f620',  # angry
        ':-@': u'\U0001f620',  # angry
        ':P': u'\U0001f61c',  # angry
        ':p': u'\U0001f61c',  # angry
        ':-P': u'\U0001f61c',  # angry
        ':-p': u'\U0001f61c',  # angry
        '-_-': u'\U0001f611',  # poker face/expressionless face
        ':*': u'\U0001f618',  # kiss
        ':-*': u'\U0001f618',  # kiss
        '=D': u'\U0001f602',  # laugh
        ':D': u'\U0001f602',  # laugh
        ';D': u'\U0001f602',  # laugh
        ';d': u'\U0001f602',  # laugh
        ':-D': u'\U0001f602',  # laugh
        'xD': u'\U0001f606',  # laugh
        'x-D': u'\U0001f606',  # laugh
        'XD': u'\U0001f606',  # laugh
        'X-D': u'\U0001f606',  # laugh

        '^_^': u'\U0000263a',  # smile

        ":')": u'\U0000263a',  # smiling face
        ":'-)": u'\U0000263a',  # smiling face
        ":'‑)": u'\U0000263a',  # smiling face

        ":'-(": u'\U0001f622',  # crying face
        ":'‑(": u'\U0001f622',  # crying face
        ":'(": u'\U0001f622',  # crying face

        ':-O': u'\U0001f632',  # surprised face
        ':O': u'\U0001f632',  # surprised face
        ':o': u'\U0001f632',  # surprised face
        ':-o': u'\U0001f632',  # surprised face

        ';)': u'\U0001f609',  # wink face
        ';-)': u'\U0001f609',  # wink face

        '<3': u'\U00002764',  # heart
        '</3': u'\U0001f494',  # broken heart

        ':-S': u'\U0001f615',  # confused face
        ':-s': u'\U0001f615',  # confused face
        ':S': u'\U0001f615',  # confused face
        ':s': u'\U0001f615',  # confused face
    }
    for k, v in replace_emoji.items():
        sent = sent.replace(k, v)
    new_sent = sent
    for c in sent:
        if c in emoji.UNICODE_EMOJI:
            new_sent = re.sub(c, ' ' + c + ' ', new_sent)
    new_sent = re.sub(' +', ' ', new_sent)
    return new_sent


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    u1 = []
    u2 = []
    u3 = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '
                line = cSpace.join(lineSplit)
            # :))) -> :)
            # :((( -> :(
            repeatedChars = [')', '(', '/']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = c + ' '
                line = cSpace.join(lineSplit)

            line = preprocessText(line)
            line = line.strip().split('\t')

            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)

            conv = ' <eos> '.join(line[1:4])

            u1.append(line[1])
            u2.append(line[2])
            u3.append(line[3])

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r' +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)

            indices.append(int(line[0]))
            conversations.append(conv.lower())

    if mode == "train":
        return indices, conversations, labels, u1, u2, u3
    else:
        return indices, conversations, u1, u2, u3


def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))

    truePositives = np.sum(discretePredictions * ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground - discretePredictions, 0, 1), axis=0)

    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision) / (macroPrecision + macroRecall) if (
                                                                                             macroPrecision + macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (
        macroPrecision, macroRecall, macroF1))

    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    print(
        "Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall) if (
                                                                                             microPrecision + microRecall) > 0 else 0
    # -----------------------------------------------------

    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions == ground)

    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (
        accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


def writeNormalisedData(dataFilePath, texts):
    """Write normalised data to a file
    Input:
        dataFilePath : Path to original train/test file that has been processed
        texts : List containing the normalised 3 turn conversations, separated by the <eos> tag.
    """
    normalisedDataFilePath = dataFilePath.replace(".txt", "_normalised.txt")
    with io.open(normalisedDataFilePath, 'w', encoding='utf8') as fout:
        with io.open(dataFilePath, encoding='utf8') as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line = line.strip().split('\t')
                normalisedLine = texts[lineNum].strip().split('<eos>')
                fout.write(line[0] + '\t')
                # Write the original turn, followed by the normalised version of the same turn
                fout.write(line[1] + '\t' + normalisedLine[0] + '\t')
                fout.write(line[2] + '\t' + normalisedLine[1] + '\t')
                fout.write(line[3] + '\t' + normalisedLine[2] + '\t')
                try:
                    # If label information available (train time)
                    fout.write(line[4] + '\n')
                except:
                    # If label information not available (test time)
                    fout.write('\n')


def getEmbeddingMatrix(wordIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}
    emoji_model = gensim.models.KeyedVectors.load_word2vec_format('./emoji2vec.txt', binary=False)

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
        elif word in emoji_model:
            print('Found emoji vec for ', word)
            # assert word in emoji.UNICODE_EMOJI
            embeddingMatrix[i] = emoji_model[word]
        else:
            embeddingMatrix[i] = np.random.uniform(0, 1, 300)
            no_emb.append(word)

    no_emb = list(set(no_emb))
    with io.open(os.path.join(gloveDir, 'no_emb.txt'), encoding="utf8", mode='w') as output:
        output.writelines('\n'.join(no_emb))

    return embeddingMatrix


#
# global trainDataPath, testDataPath, solutionPath, gloveDir
# global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
# global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE

trainDataPath = data_path + "train.txt"
# testDataPath = data_path + "devwithoutlabels.txt"
testDataPath = data_path + "testwithoutlabels.txt"
# Output file that will be generated. This file can be directly submitted.
solutionPath = "test.txt"
# Path to directory where GloVe file is saved.
gloveDir = './'

NUM_FOLDS = 2  # Value of K in K-fold Cross Validation
NUM_CLASSES = 4  # Number of classes - Happy, Sad, Angry, Others
MAX_NB_WORDS = 35000  # To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer
MAX_SEQUENCE_LENGTH = 100  # All sentences having lesser number of words than this will be padded
EMBEDDING_DIM = 300  # The dimension of the word embeddings
BATCH_SIZE = 200  # The batch size to be chosen for training the model.
LSTM_DIM = 300  # The dimension of the representations learnt by the LSTM model
DROPOUT = 0.2
LEARNING_RATE = 0.001  # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = 10

print("Processing training data...")
trainIndices, trainTexts, labels, u1_train, u2_train, u3_train = preprocessData(trainDataPath, mode="train")
# Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable   
writeNormalisedData(trainDataPath, trainTexts)
print("Processing test data...")
testIndices, testTexts, u1_test, u2_test, u3_test = preprocessData(testDataPath, mode="test")
writeNormalisedData(testDataPath, testTexts)

print("Extracting tokens...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(u1_train + u2_train + u3_train + u1_test + u2_test + u3_test)
u1_trainSequences, u2_trainSequences, u3_trainSequences = tokenizer.texts_to_sequences(
    u1_train), tokenizer.texts_to_sequences(u2_train), tokenizer.texts_to_sequences(u3_train)

u1_testSequences, u2_testSequences, u3_testSequences = tokenizer.texts_to_sequences(
    u1_test), tokenizer.texts_to_sequences(u2_test), tokenizer.texts_to_sequences(u3_test)

wordIndex = tokenizer.word_index
print("Found %s unique tokens." % len(wordIndex))

print("Populating embedding matrix...")
embeddingMatrix = getEmbeddingMatrix(wordIndex)

u1_data = pad_sequences(u1_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
u2_data = pad_sequences(u2_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
u3_data = pad_sequences(u3_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))

u1_test = pad_sequences(u1_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
u2_test = pad_sequences(u2_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
u3_test = pad_sequences(u3_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
print("Shape of training data tensor: ", u1_data.shape)
print("Shape of label tensor: ", labels.shape)

sad_ind = []
angry_ind = []
happy_ind = []
others_ind = []

for i in range(labels.shape[0]):
    if labels[i][0] == 1.:
        others_ind.append(i)
    elif labels[i][1] == 1.:
        happy_ind.append(i)
    elif labels[i][2] == 1.:
        sad_ind.append(i)
    elif labels[i][3] == 1.:
        angry_ind.append(i)

print(len(others_ind))
print(len(happy_ind))
print(len(sad_ind))
print(len(angry_ind))

others_p = np.random.permutation(others_ind)
happy_p = np.random.permutation(happy_ind)
sad_p = np.random.permutation(sad_ind)
angry_p = np.random.permutation(angry_ind)

dev_others = others_p[:2408]
dev_happy = happy_p[:106]
dev_sad = sad_p[:120]
dev_angry = angry_p[:121]

train_others = others_p[2408:]
train_happy = happy_p[106:]
train_sad = sad_p[120:]
train_angry = angry_p[121:]

dev_ind = np.concatenate([dev_others, dev_happy, dev_sad, dev_angry])
train_ind = np.concatenate([train_others, train_happy, train_sad, train_angry])

u1_train = u1_data[train_ind]
u2_train = u2_data[train_ind]
u3_train = u3_data[train_ind]
u1_dev = u1_data[dev_ind]
u2_dev = u2_data[dev_ind]
u3_dev = u3_data[dev_ind]
# dm1_train = dm1_data[train_ind]
# dm2_train = dm2_data[train_ind]
# dm3_train = dm3_data[train_ind]
# dm1_dev = dm1_data[dev_ind]
# dm2_dev = dm2_data[dev_ind]
# dm3_dev = dm3_data[dev_ind]
train_labels = labels[train_ind]
dev_labels = labels[dev_ind]

u1_train_mask = []
u2_train_mask = []
u3_train_mask = []
u1_dev_mask = []
u2_dev_mask = []
u3_dev_mask = []
u1_test_mask = []
u2_test_mask = []
u3_test_mask = []

for i in u1_train:
    train_mask = []
    for j in i:
        if j == 0:
            train_mask.append(0.)
        else:
            train_mask.append(1.)
    u1_train_mask.append(train_mask)

for i in u2_train:
    train_mask = []
    for j in i:
        if j == 0:
            train_mask.append(0.)
        else:
            train_mask.append(1.)
    u2_train_mask.append(train_mask)

for i in u3_train:
    train_mask = []
    for j in i:
        if j == 0:
            train_mask.append(0.)
        else:
            train_mask.append(1.)
    u3_train_mask.append(train_mask)

for i in u1_dev:
    train_mask = []
    for j in i:
        if j == 0:
            train_mask.append(0.)
        else:
            train_mask.append(1.)
    u1_dev_mask.append(train_mask)

for i in u2_dev:
    train_mask = []
    for j in i:
        if j == 0:
            train_mask.append(0.)
        else:
            train_mask.append(1.)
    u2_dev_mask.append(train_mask)

for i in u3_dev:
    train_mask = []
    for j in i:
        if j == 0:
            train_mask.append(0.)
        else:
            train_mask.append(1.)
    u3_dev_mask.append(train_mask)

for i in u1_test:
    train_mask = []
    for j in i:
        if j == 0:
            train_mask.append(0.)
        else:
            train_mask.append(1.)
    u1_test_mask.append(train_mask)

for i in u2_dev:
    train_mask = []
    for j in i:
        if j == 0:
            train_mask.append(0.)
        else:
            train_mask.append(1.)
    u2_test_mask.append(train_mask)

for i in u3_dev:
    train_mask = []
    for j in i:
        if j == 0:
            train_mask.append(0.)
        else:
            train_mask.append(1.)
    u3_test_mask.append(train_mask)

u1_train_mask = np.array(u1_train_mask)
u2_train_mask = np.array(u2_train_mask)
u3_train_mask = np.array(u3_train_mask)
u1_dev_mask = np.array(u1_dev_mask)
u2_dev_mask = np.array(u2_dev_mask)
u3_dev_mask = np.array(u3_dev_mask)
u1_test_mask = np.array(u1_test_mask)
u2_test_mask = np.array(u2_test_mask)
u3_test_mask = np.array(u3_test_mask)

# pickle.dump(dm1_train, open(data_path + 'dm1_train.p', 'wb'))
# pickle.dump(dm2_train, open(data_path + 'dm2_train.p', 'wb'))
# pickle.dump(dm3_train, open(data_path + 'dm3_train.p', 'wb'))
# pickle.dump(dm1_dev, open(data_path + 'dm1_dev.p', 'wb'))
# pickle.dump(dm2_dev, open(data_path + 'dm2_dev.p', 'wb'))
# pickle.dump(dm3_dev, open(data_path + 'dm3_dev.p', 'wb'))
pickle.dump(u1_train, open(data_path + 'u1_train.p', 'wb'))
pickle.dump(u2_train, open(data_path + 'u2_train.p', 'wb'))
pickle.dump(u3_train, open(data_path + 'u3_train.p', 'wb'))

pickle.dump(u1_dev, open(data_path + 'u1_dev.p', 'wb'))
pickle.dump(u2_dev, open(data_path + 'u2_dev.p', 'wb'))
pickle.dump(u3_dev, open(data_path + 'u3_dev.p', 'wb'))

pickle.dump(u1_test, open(data_path + 'u1_test.p', 'wb'))
pickle.dump(u2_test, open(data_path + 'u2_test.p', 'wb'))
pickle.dump(u3_test, open(data_path + 'u3_test.p', 'wb'))

pickle.dump(train_labels, open(data_path + 'train_labels.p', 'wb'))
pickle.dump(dev_labels, open(data_path + 'dev_labels.p', 'wb'))

pickle.dump(u1_train_mask, open(data_path + 'u1_train_mask.p', 'wb'))
pickle.dump(u2_train_mask, open(data_path + 'u2_train_mask.p', 'wb'))
pickle.dump(u3_train_mask, open(data_path + 'u3_train_mask.p', 'wb'))
pickle.dump(u1_dev_mask, open(data_path + 'u1_dev_mask.p', 'wb'))
pickle.dump(u2_dev_mask, open(data_path + 'u2_dev_mask.p', 'wb'))
pickle.dump(u3_dev_mask, open(data_path + 'u3_dev_mask.p', 'wb'))
pickle.dump(u1_test_mask, open(data_path + 'u1_test_mask.p', 'wb'))
pickle.dump(u2_test_mask, open(data_path + 'u2_test_mask.p', 'wb'))
pickle.dump(u3_test_mask, open(data_path + 'u3_test_mask.p', 'wb'))

pickle.dump(embeddingMatrix, open(data_path + 'GloVe_Embeddings.p', 'wb'))
