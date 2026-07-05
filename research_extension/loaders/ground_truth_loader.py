
import ast
import os

import pandas as pd


GROUND_TRUTH_FILE = (
    "data/latest_db/moalmanac_fda_core_query__2025-10-03.csv"
)


def load_ground_truth(path):
    """
    Load the ground truth CSV.
    """
    return pd.read_csv(path)


def extract_ground_truth(df):
    """
    Convert one row with multiple therapies
    into multiple rows with one therapy each.
    """

    rows = []

    for _, record in df.iterrows():

        therapies = ast.literal_eval(record["therapy"])

        biomarkers = ast.literal_eval(record["biomarker"])

        for therapy in therapies:

            rows.append({

                "statement_id": record["statement_id"],

                "disease":
                    record["modified_standardized_cancer"],

                "therapy":
                    therapy,

                "biomarkers":
                    ", ".join(biomarkers),

                "prompt":
                    record["prompt"]

            })

    return pd.DataFrame(rows)


def main():

    gt = load_ground_truth(GROUND_TRUTH_FILE)

    gt_df = extract_ground_truth(gt)

    os.makedirs("research_extension/data", exist_ok=True)

    output_path = "research_extension/data/ground_truth.csv"

    gt_df.to_csv(output_path, index=False)

    print("=" * 60)
    print("Ground Truth Extraction Complete")
    print("=" * 60)

    print()

    print(gt_df.head())

    print()

    print("Rows :", len(gt_df))
    print("Unique Queries :", gt_df["statement_id"].nunique())
    print("Unique Drugs :", gt_df["therapy"].nunique())

    print()

    print("Saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
