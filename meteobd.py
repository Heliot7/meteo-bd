from typing import Dict, List, Any

import click

from parse_meteo_data import parse_meteo_data, WeatherObservation
from parse_patient_data import parse_patient_data, Consultation

PATH_PATIENT_DATA = "data/dataurg.xlsx"
PATH_METEO_DATA = "data/Dades_meteorol_giques_de_la_XEMA_X4.csv"
PATH_METEO_KEYWORDS = "data/Metadades_variables_meteorol_giques.csv"


def assign_weather_to_patient(
    consultations: Dict[str, Consultation],
    weather_observations: Dict[str, Dict[str, WeatherObservation]],
) -> Dict[str, Consultation]:
    return consultations


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
    patients = parse_patient_data(PATH_PATIENT_DATA)
    weather_observations = parse_meteo_data(
        PATH_METEO_DATA, PATH_METEO_KEYWORDS, verbose
    )
    patients = assign_weather_to_patient(patients, weather_observations)
    # Dataset preparation (tensors to GPU)
    # Load NN/Model
    # Train/Test


if __name__ == "__main__":
    meteo_bd()
