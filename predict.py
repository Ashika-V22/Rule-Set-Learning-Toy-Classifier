
from dataset import load_dataset
from sequential_covering import sequential_covering, covers


def predict(input_data):

    df = load_dataset()

    rules = sequential_covering(df)

    for i, item in enumerate(rules, start=1):

        if covers(item["rule"], input_data):

            conditions = " AND ".join(
                f"{feature} = {value}"
                for feature, value in item["rule"].items()
            )

            return {
                "prediction": "Yes",
                "rule": (
                    f"Rule {i}: IF {conditions} "
                    f"THEN play = Yes"
                )
            }

    return {
        "prediction": "No",
        "rule": "No learned YES rule matched. Default rule applied: IF no learned rule matches, THEN play = NO."
    }


if __name__ == "__main__":

    sample = {
        "outlook": "Sunny",
        "temp": "Cool",
        "humidity": "Normal",
        "wind": "Weak"
    }

    result = predict(sample)

    print("\nPREDICTION RESULT")
    print("=" * 45)

    print("Input:", sample)
    print("Predicted Class:", result["prediction"])
    print("Applied Rule:", result["rule"])
