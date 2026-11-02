
import matplotlib.pyplot as plt

from dataset import load_dataset
from sequential_covering import sequential_covering
from evaluation import evaluate_rules


df = load_dataset()
rules = sequential_covering(df)
results = evaluate_rules(df, rules)


# Graph 1: Rule Coverage
plt.figure(figsize=(8, 5))

plt.bar(
    results["Rule"],
    results["Records Covered"]
)

plt.title("Number of Records Covered by Each Rule")
plt.xlabel("Classification Rules")
plt.ylabel("Records Covered")

plt.tight_layout()
plt.savefig("rule_coverage.png")
plt.show()


# Graph 2: Rule Precision
plt.figure(figsize=(8, 5))

plt.bar(
    results["Rule"],
    results["Precision (%)"]
)

plt.title("Precision of Generated Rules")
plt.xlabel("Classification Rules")
plt.ylabel("Precision (%)")
plt.ylim(0, 110)

plt.tight_layout()
plt.savefig("rule_precision.png")
plt.show()


# Graph 3: Class Distribution
positive = len(df[df["play"] == "Yes"])
negative = len(df[df["play"] == "No"])

plt.figure(figsize=(7, 5))

plt.bar(
    ["Play Tennis: Yes", "Play Tennis: No"],
    [positive, negative]
)

plt.title("Dataset Class Distribution")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()


print("\nAll three visualisations generated successfully!")
