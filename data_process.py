import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


def get_data(path='data'):
    data = []
    for d in os.listdir(path):
        if d.endswith(".csv"):
            df = pd.read_csv(path+"/"+d)
            data.append((d.split('.')[0], df))
    return data


# Specify the path to the directory containing the data folders
def get_train_test_data(directory_path='data_train_test'):

    data_folders = []

    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)
        if os.path.isdir(folder_path):
            train_folder = os.path.join(folder_path, folder_name + '_train')
            test_folder = os.path.join(folder_path, folder_name + '_test')

            train_files = [os.path.join(train_folder, file)
                           for file in os.listdir(train_folder)]
            test_files = [os.path.join(test_folder, file)
                          for file in os.listdir(test_folder)]

            train_data = []
            for tr_file in train_files:
                tr_data = pd.read_csv(tr_file, index_col=0).astype(str)
                tr_data = tr_data[~(tr_data == 'c').all(axis=1)]
                tr_data = tr_data.reset_index(drop=True)
                train_data.append(tr_data)

            test_data = []
            for te_file in test_files:
                te_data = pd.read_csv(te_file, index_col=0).astype(str)
                te_data = te_data[~(te_data == 'c').all(axis=1)]
                te_data = te_data.reset_index(drop=True)
                test_data.append(te_data)

            cross_val_data = []
            for train_file, test_file in zip(train_files, test_files):
                train_read = pd.read_csv(train_file, index_col=0).astype(str)
                train = train_read[~(train_read == 'c').all(axis=1)]
                train = train.reset_index(drop=True)
                test_read = pd.read_csv(test_file, index_col=0).astype(str)
                test = test_read[~(test_read == 'c').all(axis=1)]
                test = test.reset_index(drop=True)
                cross_val_data.append([train, test])

            # Concatenate train and test data
            data = pd.concat(cross_val_data[0], ignore_index=False)
            data = data.reset_index(drop=True)
            # data.sort_index(inplace=True)  # Sort based on the index

            data_folder = {
                'name': folder_name,
                'data': data,
                'train': train_data,
                'test': test_data,
                'cross_val': cross_val_data
            }

            data_folders.append(data_folder)

    return data_folders


def generate_fold_data(folds=5, data_folder="data"):
    csv_files = [file for file in os.listdir(
        data_folder) if file.endswith(".csv")]

    for csv_file in csv_files:
        csv_path = os.path.join(data_folder, csv_file)
        data = pd.read_csv(csv_path)
        data_features = data.iloc[:, :-1]
        data_target = data[[list(data.columns)[-1]]]

        # Create a subfolder for each CSV file
        subfolder_name = csv_file.split(".csv")[0]
        subfolder_path = os.path.join(data_folder, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        # Perform five-fold cross-validation
        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        print(subfolder_name)
        for fold, (train_index, test_index) in enumerate(kf.split(data_features, data_target)):
            fold_subfolder_path = os.path.join(
                subfolder_path, f"fold_{fold + 1}")
            os.makedirs(fold_subfolder_path, exist_ok=True)

            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]

            train_file_path = os.path.join(
                fold_subfolder_path, subfolder_name+"_train_"+str(fold + 1)+".csv")
            test_file_path = os.path.join(
                fold_subfolder_path, subfolder_name+"_test_"+str(fold + 1)+".csv")
            train_data.to_csv(train_file_path, index=False)
            test_data.to_csv(test_file_path, index=False)

        full_data_file_path = os.path.join(
            subfolder_path, subfolder_name+".csv")
        data.to_csv(full_data_file_path, index=False)


def get_full_data(directory_path='data'):

    data_folders = []

    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)
        if os.path.isdir(folder_path):
            dataset_files = [file for file in os.listdir(
                folder_path) if file.endswith('.csv')]

            dataset_file = dataset_files[0]
            dataset_data = pd.read_csv(os.path.join(
                folder_path, dataset_file)).astype(str)

            folds_data = []
            for fold_number in range(1, 6):
                fold_name = f'fold_{fold_number}'
                fold_train_file = os.path.join(
                    folder_path, fold_name, f'{dataset_file.split(".csv")[0]}_train_{fold_number}.csv')
                fold_test_file = os.path.join(
                    folder_path, fold_name, f'{dataset_file.split(".csv")[0]}_test_{fold_number}.csv')

                fold_train_data = pd.read_csv(fold_train_file).astype(str)
                fold_test_data = pd.read_csv(fold_test_file).astype(str)

                fold_data = {
                    'fold_name': fold_number,
                    'train': fold_train_data,
                    'test': fold_test_data
                }

                folds_data.append(fold_data)

            data_folder = {
                'name': folder_name,
                'data': dataset_data,
                'folds': folds_data
            }

            data_folders.append(data_folder)

    return data_folders
