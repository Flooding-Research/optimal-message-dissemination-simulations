# Simulations for Asymptotically Optimal Message Dissemination with Applications to Blockchains
This repository contains the necessary source code to reproduce the simulation results presented in the paper [Asymptotically Optimal Message Dissemination with Applications to Blockchains](https://eprint.iacr.org/2022/1723).
The Python script `sim.py` can reproduce all the data underlying the probabilistic simulations shown in the paper.

## Prerequisites
To run the simulation script, ensure you have Python 3.x installed with the following packages:

- NumPy
- tqdm 

These can be installed via `pip` by running: 

```
pip install numpy tqdm
``` 

Additionally, one must ensure that the folder in the variable `RESULT_PATH` exists. By default, this folder is set to `results`.
By adjusting the variable `NUMBER_OF_CORES` the number of cores that the simulation script should utilize can be adjusted. 

## Usage

The probabilistic simulations come with a very simple command line interface allowing to specify the number of repetitions for each data point and the figure which data is desired for. 

To run the probabilistic simulations go to the folder where `sim.py` is located and execute the command:

```
python sim.py <number_of_repetitions> <figure_number>
```

where `<number_of_repetitions>` should be replaced with the desired number of repetitions for each data point and `<figure_number>` should be replaced with the desired figure number from the [ePrint version of the paper](https://eprint.iacr.org/2022/1723) (a map between figure numbers appearing in the ePrint version and the Eurocrypt version can be found below). The latter should be in an integer between 1 and 10. 

Note that the same experiments produce the underlying data for several figures. Therefore the following figure numbers produce the same files:

- 1 and 10
- 2, 3, 4, and 5
- 6 and 7

## Mapping between ePrint and Eurocrypt figure numbers

Below is a mapping between the figure numbers. 

| Eurocrypt | ePrint |
|-----------|--------|
| 1         | 1      |
| 2         | 10     |

All other figures appearing in the ePrint version do not appear in the Eurocrypt version.
