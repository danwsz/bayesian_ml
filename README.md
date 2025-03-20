# Bayesian-ML---MVA

# PAC-Bayes VAE: Statistical Guarantees for Variational Autoencoders

## Overview
This repository contains an implementation of the code from the paper **"Statistical Guarantees for Variational Autoencoders using PAC-Bayesian Theory"**.

We have based our work on the original implementation provided by the authors at: [https://github.com/diarra2339/pac-bayes-vae](https://github.com/diarra2339/pac-bayes-vae) and have carefully adjusted it to incorporate improvements.

## Contribution
Our main contribution is the implementation of an enhanced approach using the **mixture prior**. We have restructured and modified the original code to better capture the benefits of this approach.

The **`contribution/`** folder contains the same files as the original implementation, but with our improvements incorporated.

## Installation & Usage
To run this implementation, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pac-bayes-vae-enhanced.git
   cd pac-bayes-vae-enhanced
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train.py --config config.yaml
   ```
   
For more details on how to modify parameters and test different configurations, check the documentation inside the `contribution/` folder.

## References
If you use this code, please cite the original paper and repository:

- **Paper:** "Statistical Guarantees for Variational Autoencoders using PAC-Bayesian Theory"
- **Original Code:** [https://github.com/diarra2339/pac-bayes-vae](https://github.com/diarra2339/pac-bayes-vae)

## License
This project follows the same licensing terms as the original repository.

## Contact
For any questions or contributions, feel free to open an issue or submit a pull request.
