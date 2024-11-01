import os
import sys
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('future.no_silent_downcasting', True)

def main():
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)["preprocessing"]

    size_split = params["split"]
    seed = params["seed"]

    input_file = sys.argv[1]

    output_train = os.path.join("data", "prepared", "train.csv")
    output_test = os.path.join("data", "prepared", "test.csv")

    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

    df = pd.read_csv(input_file)
    df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})

    df = df.dropna(subset='Age')
    features = ['Sex', 'Age', 'Fare', 'Survived']
    df = df[features]

    train, test = train_test_split(df, test_size=size_split, random_state=seed)

    train.to_csv(output_train)
    test.to_csv(output_test)

if __name__ == "__main__":
    main()
