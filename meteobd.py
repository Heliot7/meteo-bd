from typing import Dict, List, Any, Set

import click
import numpy as np
import torch

from network import gogo_network
from parse_meteo_data import (
    parse_meteo_data,
    WeatherObservation,
    LIST_OF_WEATHER_OBSERVATIONS,
)
from parse_patient_data import parse_patient_data, Consultation

PATH_PATIENT_DATA = "data/dataurg.xlsx"
PATH_METEO_DATA = "data/Dades_meteorol_giques_de_la_XEMA_X4.csv"
PATH_METEO_KEYWORDS = "data/Metadades_variables_meteorol_giques.csv"
MAX_DATE = "20201231"
NUM_DAYS = 7  # How many days are seen taken as reference (1 day = only same day, 7 days = the whole week)


def create_dataset_weather_to_consultation(
    weather_observations: Dict[str, Dict[str, WeatherObservation]],
    consultations: Dict[str, Consultation],
    list_discharges: List[str],
):
    number_days = len(weather_observations)
    keys = list(weather_observations.keys())[NUM_DAYS - 1 :]
    observations = list(weather_observations.values())
    data = np.zeros(
        shape=(
            number_days - (NUM_DAYS - 1),
            len(LIST_OF_WEATHER_OBSERVATIONS) * NUM_DAYS,
        )
    )
    for idx in range(len(observations)):
        if idx < NUM_DAYS - 1:
            continue
        idx_data = idx - (NUM_DAYS - 1)
        for idx_day in range(NUM_DAYS):
            idx_observation = 0
            while idx_observation < len(LIST_OF_WEATHER_OBSERVATIONS):
                data[idx_data, idx_observation * NUM_DAYS + idx_day] = observations[
                    idx - idx_day
                ][LIST_OF_WEATHER_OBSERVATIONS[idx_observation]].value
                idx_observation += 1
    labels = np.zeros([number_days - (NUM_DAYS - 1), len(list_discharges)])
    for date, consultation in consultations.items():
        # Search for date and increase by 1
        date = date.split("_")[0]
        if date in keys:
            idx_date = keys.index(date)
            list_discharges.index(consultation.discharge)
            idx_label = list_discharges.index(consultation.discharge)
            labels[idx_date, idx_label] += 1
    print("Samples size per class:")
    for idx_discharge, type_discharge in enumerate(list_discharges):
        print(f"{type_discharge}: {sum(labels[:,idx_discharge])} instances")
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
    patients, _, list_discharges = parse_patient_data(
        PATH_PATIENT_DATA, MAX_DATE, verbose
    )
    weather_observations = parse_meteo_data(
        PATH_METEO_DATA, PATH_METEO_KEYWORDS, MAX_DATE, verbose
    )
    data, labels = create_dataset_weather_to_consultation(
        weather_observations, patients, list_discharges
    )
    gogo_network(data, labels, NUM_DAYS)


if __name__ == "__main__":
    meteo_bd()
