import argparse
import twitterData
import tfscripts
import tensorflow as tf
import pandas as pd




def main(argv):

    # Fetch the data

    (train_x
     , train_y) = twitterData.load_data()

    my_feature_columns = [
        tf.feature_column.numeric_column(key='friend_count'),
        tf.feature_column.numeric_column(key='follower_count'),
        tf.feature_column.numeric_column(key='verified'),
        tf.feature_column.numeric_column(key='status_count')
    ]

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=tfscripts.my_model,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 2 classes.
            'n_classes': 2,
        })

    classifier.train(
        input_fn=lambda:tfscripts.train_input_fn(train_x, train_y, 100),
        steps=1000)

    ##THIS PIECE DOES SOME CHECKING
    predict_x = {
        'friend_count': [25, 105, 200],
        'follower_count': [1, 26, 1025],
        'verified': [1, 0, 1],
        'status_count': [23, 0, 279]
    }

    expected = {'NOT','BOT','NOT'}

    predictions = classifier.predict(
        input_fn=lambda: tfscripts.eval_input_fn(predict_x, batch_size=100))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(template.format(twitterData.CLASSIFICATION[class_id], 100 * probability, expec))



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


