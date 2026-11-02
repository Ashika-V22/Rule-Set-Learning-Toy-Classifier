
from itertools import combinations, product
from dataset import load_dataset


def covers(rule, row):
    """Check whether a record satisfies all rule conditions."""
    return all(row[feature] == value
               for feature, value in rule.items())


def sequential_covering(df, target="play", positive_class="Yes"):

    # Exclude target and identifier columns
    features = [
        col for col in df.columns
        if col not in [target, "day"]
    ]

    # Positive examples not yet covered
    remaining = df[df[target] == positive_class].copy()

    # All negative examples
    negatives = df[df[target] != positive_class]

    learned_rules = []

    while not remaining.empty:

        best_rule = None
        best_quality = -1
        best_coverage = 0

        # Generate candidate rules with 1 to 3 conditions
        for size in range(1, min(3, len(features)) + 1):

            for feature_group in combinations(features, size):

                value_lists = [
                    df[feature].unique()
                    for feature in feature_group
                ]

                for values in product(*value_lists):

                    candidate = dict(zip(feature_group, values))

                    positive_matches = remaining[
                        remaining.apply(
                            lambda row: covers(candidate, row), axis=1
                        )
                    ]

                    positive_count = len(positive_matches)

                    if positive_count == 0:
                        continue

                    negative_matches = negatives[
                        negatives.apply(
                            lambda row: covers(candidate, row), axis=1
                        )
                    ]

                    negative_count = len(negative_matches)

                    quality = positive_count / (
                        positive_count + negative_count
                    )

                    if (quality > best_quality or
                        (quality == best_quality and
                         positive_count > best_coverage)):

                        best_quality = quality
                        best_coverage = positive_count
                        best_rule = candidate

        if best_rule is None:
            break

        all_covered = df[
            df.apply(
                lambda row: covers(best_rule, row), axis=1
            )
        ]

        true_positives = len(
            all_covered[all_covered[target] == positive_class]
        )

        false_positives = len(
            all_covered[all_covered[target] != positive_class]
        )

        learned_rules.append({
            "rule": best_rule,
            "covered_records": list(all_covered.index),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "quality": (
                true_positives /
                (true_positives + false_positives)
            )
        })

        # Remove covered positive examples
        remaining = remaining.drop(
            remaining[
                remaining.apply(
                    lambda row: covers(best_rule, row), axis=1
                )
            ].index
        )

    return learned_rules


if __name__ == "__main__":

    df = load_dataset()
    rules = sequential_covering(df)

    print("\nSEQUENTIAL COVERING - GENERATED RULES")
    print("=" * 55)

    for i, item in enumerate(rules, start=1):

        conditions = " AND ".join(
            f"{feature} = {value}"
            for feature, value in item["rule"].items()
        )

        print(f"\nRULE {i}")
        print(f"IF {conditions}")
        print("THEN play = Yes")
        print("Covered record IDs:", item["covered_records"])
        print("True positives:", item["true_positives"])
        print("False positives:", item["false_positives"])
        print(f"Rule Quality: {item['quality']:.2%}")

    print("\nTOTAL RULES GENERATED:", len(rules))

    covered_positive = sum(
        item["true_positives"] for item in rules
    )

    total_positive = len(df[df["play"] == "Yes"])

    print("Positive examples covered:", covered_positive)
    print("Positive examples in dataset:", total_positive)
