from typing import Dict, Any, List

import pandas as pd
from pandas import DataFrame


class Patient:
    def __init__(self, patient_id, age, gender) -> None:
        self.patient_id = patient_id
        self.age = age
        self.gender = gender

    def print_patient_info(self) -> None:
        print(f"Patient {self.patient_id} [Age: {self.age}, Gender: {self.gender}]")


class Consultation:
    def __init__(
        self,
        consultation_id,
        patient,
        date,
        time,
        diagnosis_id,
        diagnosis_text,
        discharge,
    ) -> None:
        self.consultation_id = consultation_id
        self.patient = patient
        self.date = date
        self.time = time
        self.diagnosis_id = diagnosis_id
        self.diagnosis_text = diagnosis_text
        self.discharge = discharge

    def print_consultation_info(self) -> None:
        print(f"Consultation ID: {self.consultation_id}")
        print(f"-> Time: {self.date}, {self.time}h")
        print("-> ", end="")
        self.patient.print_patient_info()
        print(f"-> Diagnosis {self.diagnosis_id}: {self.diagnosis_text}")
        print(f"-> Discharge: {self.discharge}")

    @staticmethod
    def map_discharge_to_cluster(discharge: str) -> str:
        # Options discharge: 'Alta', 'Alta Domicili', 'Traslado otro Hospital', 'Ingreso en Hospital',
        # 'Alta Residencia', 'Fugado (no avisa)', 'No procede', 'Alta Voluntaria', 'Hospit. Domiciliaria', 'Evasión',
        # 'Alta Admin. no avisa', 'Alta administrativa avisa', 'Contraindicación clínica', 'Derivació posttriatge',
        # 'Posible donante'
        if discharge in [
            "Alta",
            "Alta Domicili",
            "Alta Residencia",
            "Alta administrativa avisa",
        ]:
            return "Discharge"
        elif discharge in [
            "Derivació posttriatge",
            "Contraindicación clínica",
            "Posible donante",
            "No procede",
        ]:
            return "Not Applicable"
        elif discharge in [
            "Fugado (no avisa)",
            "Evasión",
            "Alta Voluntaria",
            "Alta Admin. no avisa",
        ]:
            return "Voluntary Discharge"
        elif discharge in ["Hospit. Domiciliaria", "Ingreso en Hospital"]:
            return "Hospitalization"
        elif discharge in ["Traslado otro Hospital"]:
            # ToDo: Decidir que hacer (yo lo eliminaria del estudio)
            return "Referral to other Hospital"
        else:
            return discharge


def get_attribute_from_data(
    data: DataFrame, name_variable: str, idx_patient_to_remove=None
) -> List[str]:
    if idx_patient_to_remove is None:
        idx_patient_to_remove = []
    return [
        value
        for idx, value in enumerate(data[name_variable].tolist())
        if idx not in idx_patient_to_remove
    ]


def parse_patient_data(
    path_patient_data: str, verbose: bool = False
) -> Dict[str, Consultation]:
    consultation_data = pd.read_excel(path_patient_data, skiprows=4)
    discharges = get_attribute_from_data(
        consultation_data, "ALTA_Motiu Alta URG (Desc)"
    )
    discharges = [
        Consultation.map_discharge_to_cluster(discharge) for discharge in discharges
    ]
    idx_patients_to_remove = [
        idx for idx, discharge in enumerate(discharges) if discharge == "Not Applicable"
    ]
    discharges = [
        value
        for idx, value in enumerate(discharges)
        if idx not in idx_patients_to_remove
    ]

    patient_ids = get_attribute_from_data(
        consultation_data, "PAC_NHC", idx_patients_to_remove
    )
    patient_ages = get_attribute_from_data(
        consultation_data, "Edat del pacient", idx_patients_to_remove
    )
    patient_genders = get_attribute_from_data(
        consultation_data, "PAC_Sexe (Desc)", idx_patients_to_remove
    )
    consultation_ids = get_attribute_from_data(
        consultation_data, "PAC_Episodi", idx_patients_to_remove
    )
    consultation_dates = get_attribute_from_data(
        consultation_data, "ADMISSIÓ_Data", idx_patients_to_remove
    )
    consultation_times = get_attribute_from_data(
        consultation_data, "ADMISSIÓ_Hora", idx_patients_to_remove
    )
    diagnosis_ids = get_attribute_from_data(
        consultation_data, "ALTA_Diagnòstic Alta (Codi)", idx_patients_to_remove
    )
    diagnosis_text = get_attribute_from_data(
        consultation_data, "ALTA_Diagnòstic Alta (Desc)", idx_patients_to_remove
    )
    clinic_consultations = dict()
    for (
        consultation_id,
        p_id,
        p_age,
        p_gender,
        date,
        time,
        diagnosis_id,
        diagnosis_text,
        discharge,
    ) in zip(
        consultation_ids,
        patient_ids,
        patient_ages,
        patient_genders,
        consultation_dates,
        consultation_times,
        diagnosis_ids,
        diagnosis_text,
        discharges,
    ):
        if verbose:
            print(f"{p_id} {p_age} {p_gender}")
        yyyymmdd = f"{date.date().year}{date.date().month:02}{date.date().day:02}"
        hhmmss = time.replace(":", "")
        key_time = f"{yyyymmdd}_{hhmmss}"
        clinic_consultations[key_time] = Consultation(
            consultation_id,
            Patient(int(p_id), int(p_age), p_gender),
            date,
            time,
            diagnosis_id,
            diagnosis_text,
            discharge,
        )
    # Merge hourly to daily observations
    if verbose:
        for key_time, consultation in clinic_consultations.items():
            print(f"Consultation at {key_time}")
            consultation.print_consultation_info()
    return clinic_consultations
