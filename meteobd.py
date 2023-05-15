from typing import Dict, List, Any

import click

from parse_meteo_data import parse_meteo_data, WeatherObservation
from parse_patient_data import parse_patient_data, Consultation

PATH_PATIENT_DATA = "data/dataurg.xlsx"
PATH_METEO_DATA = "data/Dades_meteorol_giques_de_la_XEMA.csv"
PATH_METEO_KEYWORDS = "data/Metadades_variables_meteorol_giques.csv"


def assign_weather_to_patient(
    consultations: Dict[str, Consultation],
    weather_observations: Dict[str, Dict[str, WeatherObservation]],
) -> Dict[int, Consultation]:
    return consultations


@click.command()
def meteo_bd() -> None:
    weather_observations = parse_meteo_data(PATH_METEO_DATA, PATH_METEO_KEYWORDS)
    patients = parse_patient_data(PATH_PATIENT_DATA)
    patients = assign_weather_to_patient(patients, weather_observations)


if __name__ == "__main__":
    meteo_bd()
