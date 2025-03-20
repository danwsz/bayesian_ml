# Bayesian ML - MVA

# PAC-Bayes VAE: Statistical Guarantees for Variational Autoencoders

## Overview
This repository contains an implementation of the code from the paper **"Statistical Guarantees for Variational Autoencoders using PAC-Bayesian Theory"** by Sokhna Diarra Mbacke, Florence Clerc and Pascal Germain.

We have based our work on the original implementation provided by the authors at: [https://github.com/diarra2339/pac-bayes-vae](https://github.com/diarra2339/pac-bayes-vae) and have carefully adjusted it and incorporate improvements.

## Contribution
Our main contribution is the implementation of an enhanced approach using the **mixture prior**.
The **`contribution/`** folder contains the same files as the original implementation, but with our improvements incorporated.

## Installation & Usage
To run this implementation, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/danwsz/bayesian_ml.git
   ```
2. Clone a repository used by the author:
   ```bash
   !git clone https://github.com/cemanil/LNets.git lnets
   mv lnets/lnets/* bayesian_ml/lnets/
   cd bayesian_ml
   ```
2. If you want the classic version of the paper: 
   ```bash
   python main.py --config configs/vae.json
   ```
3. If you want the improved version: 
   ```bash
   cd contribution
   python main.py --config configs/vae.json
   ```

## Dan Winszman and Tiffany Zeitoun
