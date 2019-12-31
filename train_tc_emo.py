"""
Author: Gangeshwar
Description: Training file for Technique Classification.
"""

import pickle

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split

from models import model_CNN_emo, load_model

dev_template_labels_file = "datasets/dev-task-TC-template.out"

data_path = './processed_data/'


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


def load_data():
    print("Loading data...")
    train_x = pickle.load(open(data_path + 'train_x.p', 'rb'))
    train_seq_len = pickle.load(open(data_path + 'train_seq_len.p', 'rb'))
    train_y = pickle.load(open(data_path + 'train_y.p', 'rb'))
    dev_x = pickle.load(open(data_path + 'dev_x.p', 'rb'))
    dev_seq_len = pickle.load(open(data_path + 'dev_seq_len.p', 'rb'))
    train_x_emo = pickle.load(open(data_path + 'train_x_emotion.p', 'rb'))
    dev_x_emo = pickle.load(open(data_path + 'dev_x_emotion.p', 'rb'))
    GloVe_Embeddings = pickle.load(open(data_path + 'GloVe_Embeddings.p', 'rb'))
    # print(train_y)
    return train_x, train_x_emo, train_seq_len, train_y, dev_x, dev_x_emo, dev_seq_len, GloVe_Embeddings


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
if __name__ == '__main__':
    train_x, train_x_emo, train_seq_len, train_y, dev_x, dev_x_emo, dev_seq_len, embeddings = load_data()

    # Split the data
    train_x, test_x, train_x_emo, test_x_emo, train_seq_len, test_seq_len, train_y, test_y = train_test_split(train_x,
                                                                                                              train_x_emo,
                                                                                                              train_seq_len,
                                                                                                              train_y,
                                                                                                              test_size=0.10,
                                                                                                              shuffle=True,
                                                                                                              stratify=train_y)

    print(train_x_emo.shape)
    print(dev_x_emo.shape)

    # exit()

    # s = int(train_x.shape[0] * 0.90)
    # print(s, train_x.shape[0])

    # test_x = train_x[s:]
    # train_x = train_x[:s]
    #
    # test_x_emo = train_x_emo[s:]
    # train_x_emo = train_x_emo[:s]
    #
    # test_seq_len = train_seq_len[s:]
    # train_seq_len = train_seq_len[:s]
    #
    # test_y = train_y[s:]
    # train_y = train_y[:s]

    model = model_CNN_emo(embeddings)

    lr = 0.0001
    bz = 256
    epochs = 300

    opt = Adam(lr=lr)
    opt = SGD(0.01)
    # print(str(opt))
    # exit()
    model_name = 'text_Adam_lr%s_bz%s' % (lr, bz)
    model_path = 'models/%s' % (model_name)
    checkpoint = ModelCheckpoint('%s.{epoch:02d}.hdf5' % (model_path), monitor='loss', verbose=1,
                                 save_best_only=False, mode='auto')

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    try:
        model.fit([train_x, train_x_emo], train_y, validation_data=[[test_x, test_x_emo], test_y], epochs=epochs,
                  batch_size=bz,
                  shuffle=True, callbacks=[checkpoint])
    except:
        pass

    epoch = input("\n\nWhich epoch to load?\nAns: ")
    epoch = int(epoch)
    load_model_path = '%s.%02d.hdf5' % (model_path, epoch)
    print('Loading model - ', load_model_path)
    model = load_model(load_model_path, custom_objects={'loss': 'categorical_crossentropy'})
    # print(model.summary())
    predictions = model.predict([dev_x, dev_x_emo])
    predictions = predictions.argmax(axis=1)

    # writing predictions to file
    task_TC_output_file = "model-output-TC.txt"
    dev_article_ids, dev_span_starts, dev_span_ends, dev_labels = read_predictions_from_file(dev_template_labels_file)

    with open(task_TC_output_file, "w") as fout:
        for article_id, prediction, span_start, span_end in zip(dev_article_ids, predictions, dev_span_starts,
                                                                dev_span_ends):
            fout.write("%s\t%s\t%s\t%s\n" % (article_id, index2label[prediction], span_start, span_end))
    print("Predictions written to file " + task_TC_output_file)
