import pickle

from keras.optimizers import RMSprop
from tensorflow.python.ops import nn

from models import model_custom

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
    train_x = pickle.load(open(data_path + 'train_x.p', 'rb'))
    train_seq_len = pickle.load(open(data_path + 'train_seq_len.p', 'rb'))
    train_y = pickle.load(open(data_path + 'train_y.p', 'rb'))
    dev_x = pickle.load(open(data_path + 'dev_x.p', 'rb'))
    dev_seq_len = pickle.load(open(data_path + 'dev_seq_len.p', 'rb'))
    GloVe_Embeddings = pickle.load(open(data_path + 'GloVe_Embeddings.p', 'rb'))
    # print(train_y)
    return train_x, train_seq_len, train_y, dev_x, dev_seq_len, GloVe_Embeddings


def create_split():
    pass


if __name__ == '__main__':
    train_x, train_seq_len, train_y, dev_x, dev_seq_len, embeddings = load_data()

    s = int(train_x.shape[0] * 0.90)
    print(s, train_x.shape[0])

    test_x = train_x[s:]
    train_x = train_x[:s]

    test_seq_len = train_seq_len[s:]
    train_seq_len = train_seq_len[:s]

    test_y = train_y[s:]
    train_y = train_y[:s]

    model = model_custom(embeddings)

    opt = RMSprop()

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    model.fit(train_x, train_y, validation_data=[test_x, test_y], epochs=5, batch_size=256, shuffle=True)

    predictions = model.predict(dev_x)

    # writing predictions to file
    task_TC_output_file = "model-output-TC.txt"
    dev_article_ids, dev_span_starts, dev_span_ends, dev_labels = read_predictions_from_file(dev_template_labels_file)

    with open(task_TC_output_file, "w") as fout:
        for article_id, prediction, span_start, span_end in zip(dev_article_ids, predictions, dev_span_starts,
                                                                dev_span_ends):
            fout.write("%s\t%s\t%s\t%s\n" % (article_id, prediction, span_start, span_end))
    print("Predictions written to file " + task_TC_output_file)
