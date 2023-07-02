Config file:
- train/test
  - epochs
- input_params
- output_params

Experiments:
- Input data
  - How many days as historical data = {1, 3, 7}
  - Combination of Meteo parameters (3 most important)
    - Temperatura Maxima (del dia)
    - Pressio atmosferica Maxima (del dia)
    - Irradiancio solar global (total del dia - hores de sol?)
- Output data (1xN and 1xM or 2xN?)
  - Diagnosis = [XXX: 0, YYY: 1, ZZZ: 2, ...] - Episodio maniaco
  - Discharge = [XXX: 0, YYY: 1, ZZZ: 2, ...] - Ingreso


