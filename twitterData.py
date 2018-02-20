import pandas as pd

CLASSIFICATION = ['BOT', 'NOT']



CSV_COLUMN_NAMES = ['friend_count','follower_count','verified' ,'status_count','bot']


def load_data(label_name='bot'):

    train_path = '~/Documents/SEC203DAT/TrainingData.csv'

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                       )
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop(label_name)

    # Return four DataFrames.
    return (train_features, train_label)

