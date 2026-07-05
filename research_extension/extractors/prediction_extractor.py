
import pickle
import json
import pandas as pd 

PREDICTION_FILE = (
    "output/RAG_res_gpt4o/structured_latest_db/"
    "RAGstra0n1temp0.0_res_dict.pkl"
)

def load_prediction_file(path):
    """
    Load a prediction pickle file.
    """

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data

def inspect_prediction_file(data):
    """
    Print basic information about the prediction file.
    """

    print("=" * 60)
    print("Prediction File Inspection")
    print("=" * 60)

    print("\nKeys:")
    print(data.keys())

    print("\nNumber of experiment outputs:")
    print(len(data["full output"]))

    print("\nRuntime:")
    print(data["runtime"])

def extract_all_predictions(data):
    """
    Extract every treatment from every prediction.

    Returns
    -------
    pandas.DataFrame
    """

    experiment = data["full output"][0]

    rows = []

    for prediction_id, prediction in enumerate(experiment):

        prediction = json.loads(prediction)

        for treatment_name, treatment in prediction.items():

            row = {
                "prediction_id": prediction_id,
                "treatment_number": treatment_name,
                "disease": treatment.get("Disease Name"),
                "disease_phase": treatment.get("Disease Phase or Condition"),
                "drug": treatment.get("Drug Name"),
                "prior_treatment": treatment.get("Prior Treatment or Resistance Status"),
                "genomic_features": treatment.get("Genomic Features"),
                "fda_label": treatment.get("Link to FDA-approved Label")
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    return df

def main():

    data = load_prediction_file(PREDICTION_FILE)

    df = extract_all_predictions(data)

    print(df.head())

    print()

    print("="*60)

    print("Total Rows :", len(df))

    print("Unique Predictions :", df["prediction_id"].nunique())

    print("Unique Drugs :", df["drug"].nunique())

    print("="*60)

    import os

    os.makedirs("research_extension/data", exist_ok=True)

    csv_path = "research_extension/data/predictions.csv"

    df.to_csv(csv_path, index=False)

    print()
    print("CSV Saved Successfully!")
    print(csv_path) 

if __name__ == "__main__":
    main()
