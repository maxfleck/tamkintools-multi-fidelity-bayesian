# tamkintools_multi_fidelity_bayesian

#### Supporting Information / Code for the publication xxx  

This repository consists of several examples combining different levels of quantum theory with Multi-Fidelity modeling and Bayesian methods including a problem-oriented acquisition function for thermodynamic properties.

All plots shown in the paper can be reproduced with this code.

### PI_example

a starting example to get used to the approach and play around with kernels and hyperparameters

#### PI_tamkin_multi_fidelity

- first example, focus on testing different kernels
- quantum mechanic calculation results from different levels of theory are combined

#### PI_reduced_hf_data

- impact of missing (high energy) samples can be tested here


### tetrahydrofuran

in this example quantum mechanic results from different levels of theory contradict each other

#### multi_fidelity_tetrahydrofuran

- shows behavior of linear Multi-Fidelity models for contradicting samples


### Umer_results

in this example the impact of missing high energy high fidelity samples is shown

#### umer_mf

- shows the impact of missing high energy high fidelity samples
- deviates a sampling strategy for initial high fidelity samples when a low fidelity scan is available


### DMM_R3H+CH3

most extensive example including Bayesian techniques for better sampling

#### ThermoAquisition_thermoProps

- only low fidelity samples are used
- a problem oriented acquisition function is introduced and tested
- thermodynamic properties are calculated based on the samples
- ... and can be compared to our extensively sampled reference

#### ThermoAquisitionDyn

- a dynamic version of our problem oriented acquisition function is introduced
- adjusts its parameters on the run
- useful when energy range of a scan is a priori unknown

#### ThermoAquisitionMF_e1_scan1

- utilizes everything:
	- Multi-Fidelity model
	- sampling strategy for initial high fidelity samples
	- Bayesian methods 
	- our acquisition function
- allows deeper insights

#### ThermoAquisitionMF_TS_scan2

- utilizes everything:
	- Multi-Fidelity model
	- sampling strategy for initial high fidelity samples
	- Bayesian methods 
	- our acquisition function
- allows deeper insights