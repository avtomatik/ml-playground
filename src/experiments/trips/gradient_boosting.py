import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from core.paths import DATA_DIR


def main(file_name: str = "trips_data.xlsx") -> None:
    trips = pd.read_excel(DATA_DIR / file_name)
    trips_processed = pd.get_dummies(
        trips,
        columns=[
            "city",
            "vacation_preference",
            "transport_preference",
        ],
    )
    classifier = GradientBoostingClassifier()
    input_data = trips_processed.drop("target", axis=1)
    output_data = trips_processed.target
    classifier.fit(input_data, output_data)
    print({col: 0 for col in trips_processed.columns})
