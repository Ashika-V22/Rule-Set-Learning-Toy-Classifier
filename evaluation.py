
import pandas as pd
from dataset import load_dataset
from sequential_covering import sequential_covering, covers


def evaluate_rules(df, rules, target="play",
                   positive_class="Yes"):

    results = []

    for i, item in enumerate(rules, start=1):

        rule = item["rule"]

        covered = df[
            df.apply(
                lambda row: covers(rule, row), axis=1
            )
        ]

        true_positives = len(
            covered[covered[target] == positive_class]
        )

        false_positives = len(
            covered[covered[target] != positive_class]
        )

        total_covered = len(covered)

        precision = (
            true_positives / total_covered * 100
            if total_covered > 0 else 0
        )

        conditions = " AND ".join(
            f"{feature} = {value}"
            for feature, value in rule.items()
        )

        results.append({
            "Rule": f"Rule {i}",
            "Conditions": conditions,
            "Records Covered": total_covered,
            "True Positives": true_positives,
            "False Positives": false_positives,
            "Precision (%)": round(precision, 2)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    df = load_dataset()
    rules = sequential_covering(df)

    results = evaluate_rules(df, rules)

    print("\nRULE COVERAGE ANALYSIS")
    print("=" * 80)
    print(results.to_string(index=False))

    total_positive = len(df[df["play"] == "Yes"])

    covered_positive = sum(
        item["true_positives"] for item in rules
    )

    total_negative = len(df[df["play"] != "Yes"])

    incorrectly_covered = sum(
        item["false_positives"] for item in rules
    )

    print("\nOVERALL RESULTS")
    print("=" * 80)

    print("Total positive examples:", total_positive)
    print("Positive examples covered:", covered_positive)

    coverage = (
        covered_positive / total_positive * 100
        if total_positive else 0
    )

    print("Positive coverage:", f"{coverage:.2f}%")
    print("Total negative examples:", total_negative)
    print("Incorrectly covered negative examples:",
          incorrectly_covered)
