# Unitary Observer Dynamics & Decoherence

This repository contains the exact computational framework used to simulate observer dynamics, environment-induced decoherence, and macroscopic Wigner's Friend superpositions within strictly unitary quantum mechanics. 

The code serves as a reproducible numerical testbed accompanying the foundational physics paper: **"The Limits of Objective Classicality: A Computational First-Principles Study of Observer-Induced Decoherence in Unitary Quantum Mechanics."**

## Overview

While environment-induced decoherence explains the suppression of interference between macroscopically distinct states, this project computationally tests whether decoherence alone can dynamically produce a classical measurement record for a physical observer in a closed quantum system.

The simulation models a closed quantum universe comprising three subsystems:
1. **The Quantum System (Q):** A two-level qubit.
2. **The Environment (E):** A finite spin-bath.
3. **The Observer (O):** A quantum particle in a double-well potential acting as a nonlinear amplifying degree of freedom.

By avoiding master-equation approximations and utilizing exact Krylov subspace exponential integration, the simulation tracks the full unitary evolution without invoking a collapse postulate.

## Features

* **Exact Unitary Propagation:** Uses `scipy.sparse.linalg.expm_multiply` for high-precision time evolution of the global wavefunction.
* **Sparse Tensor Products:** Efficiently lifts local Hamiltonian operators to the global tensor-product Hilbert space.
* **Partial Trace Diagnostics:** Dynamically calculates reduced state purities for the qubit and the observer to track entanglement and decoherence.
* **Publication-Ready Plotting:** Automatically generates 5 high-quality, scaled, and formatted plots corresponding to the control and full unitary simulations.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/unitary-observer-dynamics.git](https://github.com/yourusername/unitary-observer-dynamics.git)
   cd unitary-observer-dynamics
