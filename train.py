import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier



def main():
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    input_path = Path().cwd() / input_path
    output_path = Path().cwd() / output_path

    os.makedirs(output_path, exist_ok=True)

    input_train = input_path / "train.pkl"
    input_test = input_path / "test.pkl"

    df_train = pd.read_pickle(input_train)
    df_test = pd.read_pickle(input_test)

    X_train = df_train.drop(columns=['Survived'])
    X_test = df_test.drop(columns=['Survived'])
    y_train = df_train['Survived']
    y_test = df_test['Survived']

    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)["train"]

    # max_depth = params['max_depth']
    # min_child_weight = params['min_child_weight']
    # eta = params['eta']
    # gamma = params['gamma']

    model_xgb = XGBClassifier(**params, early_stopping_rounds=20)
    model_xgb.fit(
        X_train, 
        y_train, 
        eval_set=[(X_test, y_test)], 
        verbose=False
    )

    model_xgb.save_model(output_path / "xgb_model.json")

    # predict_xgb = model_xgb.predict(X_test)



    # Params: 
    #     max_depth: 9
    #     min_child_weight: 3
    #     eta: 0.004658536774843258
    #     gamma: 0.0010325021635031145

if __name__ == "__main__":
    main()