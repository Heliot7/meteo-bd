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

Results: (lr=1e-4, bs=3, ep=100)
- Config: 5 variables
  - Context: 3 days
    - [15-5, 5-5, 5-4]: Accuracy: 98.0%, Avg loss: 11.499335
    - [15-5, 5-5, 5-5, 5-4]: Accuracy: 96.9%, Avg loss: 11.494569
    - [15-5, 5-4]: Accuracy: 98.0%, Avg loss: 11.524478
    - [15-4, 4-4, 4-4]: Accuracy: 96.9%, Avg loss: 11.492168
    - [15-15, 15-10, 15-4]: Accuracy: 97.1%, Avg loss: 11.494606
    - [15-15, 15-10, 10-4]: Accuracy: 96.9%, Avg loss: 11.598397

  - Context: 7 days
    - [15-5, 5-5, 5-4]: Accuracy: 98.0%, Avg loss: 11.510802
    - [15-15, 15-10, 15-4]: 98.0%, Avg loss: 11.534071
  - Context: 1 day
    - [15-5, 5-5, 5-4]: 97.9%, Avg loss: 11.510404
    - [15-15, 15-10, 10-4]: Accuracy: 96.9%, Avg loss: 11.730655
- Config: 3 variables
  - Context: 3 days
    - [15-5, 5-5, 5-4]: Accuracy: 98.0%, Avg loss: 11.522683

- Regression loss
  - [35-4, 4-4, 4-4] (same with 5)
```
Epoch 250 with lr 1.00E-8
-------------------------------
loss: 4.821068  [   25/ 3176]
loss: 3.948973  [ 2525/ 3176]
Test Error: 
Accuracy: 97.9%, Avg loss: 5.986494
Predicted: "tensor([ 1.5526,  1.6827,  0.0842, 10.8484], device='cuda:0')", Actual: "tensor([ 1.,  4.,  0., 17.], device='cuda:0')"
Predicted: "tensor([ 1.4544,  1.5621, 10.4662,  0.1652], device='cuda:0')", Actual: "tensor([ 1.,  0., 12.,  1.], device='cuda:0')"
 Accuracy: 96.2%, Avg loss: 6.050419 
Predicted: "tensor([ 1.7128,  1.6169, 10.6310, -0.1774], device='cuda:0')", Actual: "tensor([ 0.,  1., 12.,  1.], device='cuda:0')"
```

  - [35-35, 35-15, 15-4]

  - [35-4, 4-4, 4-4, 4-4] 