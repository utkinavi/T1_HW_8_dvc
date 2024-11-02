import os
import sys
import yaml
from pathlib import Path
import pandas as pd

def read_df(path):
    return pd.read_csv(path)

def generate_feature_and_save_train(path_input_train, path_save_train):

    #####################
    #featurize
    #####################

    df_trian = read_df(path_input_train)

    df_trian.to_pickle(path_save_train)


def generate_feature_and_save_test(path_input_test, path_save_test):
    
    #####################
    #featurize
    #####################

    df_test = read_df(path_input_test)

    df_test.to_pickle(path_save_test)


def main():
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)["featurize"]

    seed = params["seed"]

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    input_path = Path().cwd() / input_path
    output_path = Path().cwd() / output_path

    input_train = input_path / "train.csv"
    input_test = input_path / "test.csv"
    output_train = output_path / "train.pkl"
    output_test = output_path /  "test.pkl"

    os.makedirs(output_path, exist_ok=True)

    generate_feature_and_save_train(input_train, output_train)
    generate_feature_and_save_test(input_test, output_test)

if __name__ == "__main__":
    main()

