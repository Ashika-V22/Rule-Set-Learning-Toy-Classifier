
# Rule-Set Learning for a Toy Classifier

## Project Overview

This project demonstrates the Sequential Covering algorithm for learning IF-THEN classification rules using the Play Tennis dataset.

The application predicts whether weather conditions are suitable for playing tennis based on automatically learned rules.

## Features

- Automatic generation of IF-THEN classification rules.
- Rule coverage and precision evaluation.
- Interactive weather-based classification.
- Prediction explanations using learned rules.
- Display of experimental results.

## Dataset

**Kaggle Play Tennis Dataset**

The dataset contains 14 records with four input features:

- Outlook
- Temperature
- Humidity
- Wind

Target variable: Play (Yes/No).

## Algorithm

The Sequential Covering algorithm generates classification rules one at a time.

It selects candidate rules based on their precision and coverage of positive examples. Covered positive examples are removed, and the process continues until all positive examples are covered or no suitable rule can be generated.

## Experimental Results

Three IF-THEN rules were generated.

- Total rules: 3
- Positive examples covered: 9 out of 9
- Positive coverage: 100%
- Rule precision: 100% for all three rules

These results are based on the training dataset and do not establish performance on unseen data.

## Technologies Used

- Python
- Pandas
- Streamlit
- Matplotlib

## Project Structure

```text
Rule_Set_Learning/
│
├── data/
│   └── play_tennis.csv
│
├── dataset.py
├── sequential_covering.py
├── evaluation.py
├── visualization.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## How to Run Locally

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Limitations

- Small dataset containing only 14 records.
- Limited number of input features.
- No evaluation on unseen data.
- Greedy rule selection may not produce an optimal rule set.
- Unmatched records are classified using a default NO rule.
