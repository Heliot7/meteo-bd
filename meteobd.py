from typing import Dict, List, Any, Set

import click
import numpy as np
import torch

from parse_meteo_data import parse_meteo_data, WeatherObservation
from parse_patient_data import parse_patient_data, Consultation

PATH_PATIENT_DATA = "data/dataurg.xlsx"
PATH_METEO_DATA = "data/Dades_meteorol_giques_de_la_XEMA_X4.csv"
PATH_METEO_KEYWORDS = "data/Metadades_variables_meteorol_giques.csv"


def create_dataset_weather_to_consultation(weather_observations: Dict[str, Dict[str, WeatherObservation]], consultations: Dict[str, Consultation], list_discharges: List[str]):
    number_days = len(weather_observations)
    keys = list(weather_observations.keys())
    data = np.zeros(number_days)
    for idx, observations in enumerate(weather_observations.values()):
        data[idx] = observations["Temperatura màxima"].value
    labels = np.zeros([number_days, len(list_discharges)])
    for date, consultation in consultations.items():
        # Search for date and increase by 1
        date = date.split("_")[0]
        if date in keys:
            idx_date = keys.index(date)
            list_discharges.index(consultation.discharge)
            idx_label = list_discharges.index(consultation.discharge)
            labels[idx_date, idx_label] += 1
    return data, labels


@click.command()
@click.option(
    "--config_file", required=True, help="Config file with tran/test variables."
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    required=False,
    default=False,
    help="Print more output/plots.",
)
def meteo_bd(config_file: str, verbose: bool) -> None:
    patients, _, list_discharges = parse_patient_data(PATH_PATIENT_DATA, verbose)
    weather_observations = parse_meteo_data(
        PATH_METEO_DATA, PATH_METEO_KEYWORDS, verbose
    )
    data, labels = create_dataset_weather_to_consultation(weather_observations, patients, list_discharges)
    # Dataset preparation (tensors to GPU)
    tensor_data = torch.tensor(data)
    tensor_data = torch.tensor(labels)
    print(data)
    print(labels)
    # Load NN/Model
    # Train/Test


if __name__ == "__main__":
    meteo_bd()
