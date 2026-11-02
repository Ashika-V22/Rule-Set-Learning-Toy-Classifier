
import pandas as pd

DATASET_PATH = "data/play_tennis.csv"

def load_dataset():
    df = pd.read_csv(DATASET_PATH)

    # Remove unnecessary index columns if present
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    return df


if __name__ == "__main__":
    df = load_dataset()

    print("\nKAGGLE PLAY TENNIS DATASET")
    print("=" * 60)

    print("\nDataset Preview:")
    print(df.to_string(index=False))

    print("\nDataset Information:")
    print("Total Records:", len(df))
    print("Total Columns:", len(df.columns))
    print("Column Names:", list(df.columns))

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nClass Distribution:")
    print(df.iloc[:, -1].value_counts())
