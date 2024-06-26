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

Additionally, one must ensure that the script can write to the `RESULT_PATH`. By default, this folder is set to `results`, and it is created if it does not already exist.
By adjusting the variable `NUMBER_OF_CORES` the number of cores that the simulation script utilizes can be adjusted. By default, all available CPU cores are used.

## Usage

The probabilistic simulations come with a very simple command line interface allowing to specify the number of repetitions for each data point and the figure for which data should be simulated. 

To run the probabilistic simulations go to the folder where `sim.py` is located and execute the command:

```
python sim.py <number_of_repetitions> <figure_number>
```

where `<number_of_repetitions>` should be replaced with the desired number of repetitions for each data point and `<figure_number>` should be replaced with the desired figure number (an integer between 1 and 10) from the [ePrint version of the paper](https://eprint.iacr.org/2022/1723) (a map between figure numbers appearing in the ePrint version and the Eurocrypt version can be found [here](#figure-mapping)). We suggest to use `1000` repetitions for experimentation and `100000` repetitions to reproduce the results of the paper.

Note that the same experiments produce the underlying data for several figures. Therefore the following figure numbers (following the ePrint numbering) produce the same files and outputs:

- 1 and 10
- 2, 3, 4, and 5
- 6 and 7


## Obtaining the Figures in the Eurocrypt Version of the Paper
Figures 1 and 2 from the Eurocrypt paper correspond to figures 1 and 10 in the ePrint version. As explained above, the corresponding simulations can be run by executing the command
```
python sim.py <number_of_repetitions> 1
```
or `python sim.py <number_of_repetitions> 10`, which produces the same results. How to obtain the corresponding figures from the outputs is explained in the sections below.

### Figure 1 (Eurocrypt numbering)
The simulations produce 9 files in total where each file contains the columns `Error rate` and `Pr. party communication in MB` that correspond to the data points for a particular line in Figure 1.

The 3 files for the protocol __FFlood__ are named:
``` 
./results/FF-n-4096-r-<number_of_repetitions>.csv 
./results/FF-n-8192-r-<number_of_repetitions>.csv 
./results/FF-n-16384-r-<number_of_repetitions>.csv
```
The 6 files for the protocol __ECFlood(d)__ are named:
```
./results/FFFloodAmplifier-n-4096-d-8-mu-25-r-<number_of_repetitions>.csv 
./results/FFFloodAmplifier-n-8192-d-8-mu-25-r-<number_of_repetitions>.csv 
./results/FFFloodAmplifier-n-16384-d-8-mu-25-r-<number_of_repetitions>.csv
./results/FFFloodAmplifier-n-4096-d-20-mu-10-r-<number_of_repetitions>.csv 
./results/FFFloodAmplifier-n-8192-d-20-mu-10-r-<number_of_repetitions>.csv
./results/FFFloodAmplifier-n-16384-d-20-mu-10-r-<number_of_repetitions>.csv
```
### Figure 2 (Eurocrypt numbering)
The output files are not needed to reproduce this figure. Instead, parameters output to the terminal need to be used to plot the per-party communication complexity as a function of the message length for the respective functions.

For __FFlood__ the function to plot is:

```
FFlood_per_party_communication(msg_length) = degree * msg_length
```

For the respective number of parties, the script produces 3 outputs of the following form for __FFlood__
```
Best parameter that made FF-n-<number_of_parties> not fail is degree = <d>
```

From these outputs, the best degree ensuring that all simulations succeeded can be read and used in the above function to obtain the lines for __FFlood__ in Figure 2 (Eurocrypt numbering). 

For __ECFlood__ the function that should be plotted is (Eq (11) in the Eurocrypt version replacing the total number of shares minus the number of tolerated erasures with the total number of shares times the fraction of shares needed for reconstruction): 

```
ECFlood_per_party_communication(msg_length) = number_of_shares * degree * (ceil(msg_length / (reconstruction_fraction * number_of_shares)) + 257 * ceil(log2(number_of_shares)) + 256)
```

For the respective number of parties, the script produces 6 outputs of the following form for __ECFlood__
```
Best parameters that made FFFloodAmplifier-n-<number_of_parties>-d-<degree>-mu-<number_of_shares> not fail is reconstruction_fraction = <reconstruction_fraction>
```
From these outputs, the best reconstruction threshold ensuring that all simulations succeeded for the respective combinations of degree and number of shares, can be read and used in the above function to obtain the lines for __ECFlood__ in Figure 2 (Eurocrypt numbering). 

## Mapping between ePrint and Eurocrypt figure numbers

Below is a mapping between the figure numbers. 

| Eurocrypt | ePrint |
|-----------|--------|
| 1         | 1      |
| 2         | 10     |

All other figures appearing in the ePrint version do not appear in the Eurocrypt version.
