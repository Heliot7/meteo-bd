from typing import Dict, List

import pandas as pd
from pandas import DataFrame

CODE_STATION_CLINIC = "X4"  # El Raval (closest to Hospital Clinic BCN)


class WeatherObservation:
    def __init__(self, code, name, value, time) -> None:
        self.code = code
        self.name = name
        self.value = value
        self.time = time

    def print_weather_observation(self) -> None:
        print(f"{self.name} ({self.code}) = {self.value} at this time {self.time}")


def map_meteo_code_to_name(map_meteo_code: Dict[int, str], code: int) -> str:
    return map_meteo_code[code]


def map_time_to_std_format(datetime: str) -> str:
    datetime_split = datetime.split(" ")
    date_split = datetime_split[0].split("/")
    time_split = datetime_split[1].split(":")
    hour = time_split[0] if datetime_split[2] == "AM" else str(int(time_split[0]) + 12)
    hour = "00" if hour == "12" else hour
    hour = "12" if hour == "24" else hour
    return f"{date_split[2]}{date_split[1]}{date_split[0]}_{hour}{time_split[1]}{time_split[2]}"


def get_data_from_station(
    csv_meteo_data: DataFrame, map_meteo_code: Dict[int, str], code_station: str
) -> Dict[str, Dict[str, WeatherObservation]]:
    meteo_data_station = csv_meteo_data["CODI_ESTACIO"]
    idx_station = [
        idx for idx, value in enumerate(meteo_data_station) if value == code_station
    ]
    codes = csv_meteo_data["CODI_VARIABLE"][idx_station]
    values = csv_meteo_data["VALOR_LECTURA"][idx_station]
    times = csv_meteo_data["DATA_LECTURA"][idx_station]
    weather_observation = {}
    for code, value, time in zip(codes, values, times):
        keyword_time = map_time_to_std_format(time)
        if keyword_time not in weather_observation.keys():
            weather_observation[keyword_time] = {}
        observation_type = map_meteo_code_to_name(map_meteo_code, code)
        weather_observation[keyword_time][observation_type] = WeatherObservation(
            code, observation_type, value, time
        )
    return weather_observation


def print_time_weather_observations(
    time: str, observations: Dict[str, WeatherObservation]
) -> None:
    print(f"At {time}:")
    [observation.print_weather_observation() for observation in observations.values()]


def parse_meteo_data(
    path_meteo_data: str, path_meteo_keywords: str, verbose: bool = False
) -> Dict[str, Dict[str, WeatherObservation]]:
    csv_meteo_keywords = pd.read_csv(path_meteo_keywords)
    meteo_variables_code = csv_meteo_keywords["CODI_VARIABLE"].tolist()
    meteo_variables_name = csv_meteo_keywords["NOM_VARIABLE"].tolist()
    map_meteo_code = {
        code: name for code, name in zip(meteo_variables_code, meteo_variables_name)
    }
    csv_meteo_data = pd.read_csv(path_meteo_data)
    weather_observations = get_data_from_station(
        csv_meteo_data, map_meteo_code, CODE_STATION_CLINIC
    )
    if verbose:
        [
            print_time_weather_observations(key, observations)
            for key, observations in weather_observations.items()
        ]
    return weather_observations
