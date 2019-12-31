from keras.engine.saving import load_model

from train_tc_emo import read_predictions_from_file, load_data

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

    dev_template_labels_file = "datasets/dev-task-TC-template.out"

    train_x, train_x_emo, train_seq_len, train_y, dev_x, dev_x_emo, dev_seq_len, embeddings = load_data()
    load_model_path = input("\n\nWhich model to load?\nAns: ")
    # epoch = int(epoch)
    # load_model_path = '%s.%02d.hdf5' % (model_path, epoch)
    print('Loading model - ', load_model_path)
    model = load_model(load_model_path, custom_objects={'loss': 'categorical_crossentropy'})
    # print(model.summary())
    predictions = model.predict([dev_x])
    predictions = predictions.argmax(axis=1)

    # writing predictions to file
    task_TC_output_file = "model-output-TC.txt"
    dev_article_ids, dev_span_starts, dev_span_ends, dev_labels = read_predictions_from_file(dev_template_labels_file)

    with open(task_TC_output_file, "w") as fout:
        for article_id, prediction, span_start, span_end in zip(dev_article_ids, predictions, dev_span_starts,
                                                                dev_span_ends):
            fout.write("%s\t%s\t%s\t%s\n" % (article_id, index2label[prediction], span_start, span_end))
    print("Predictions written to file " + task_TC_output_file)
