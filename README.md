# Bidirectional Generative Adversarial Network - Entropic Path Sampling

The repository documents how to perform bidirectional generative adversarial network - entropic path sampling ([BGAN-EPS](https://pubs.acs.org/doi/10.1021/acs.jpcb.3c01202)) method.<sup>1</sup>
<p align="center">
<img src="https://user-images.githubusercontent.com/25111091/205413472-bf70e899-32f7-4a0c-8dc5-a576c129a36c.jpg" width=50%>
</p>

## Module Requirements
        python 3.9.18
        numpy 1.26.0
        pytorch 2.1.0
        networkx 3.1
        scikit-learn 1.3.0
        mdtraj 1.9.9

        Entropy Profile Graphing Module
        matplotlib 3.9.1
        scienceplots 2.1.1

## Install
BGAN-EPS can be downloaded by

        git clone https://github.com/rshin1209/bgan_eps.git

The software has been tested on Rocky Linux release 8.9 and Python 3.9 environment. CUDA is recommended for accelerating the training process.

## How to run BGAN-EPS

### Example Reaction: Diene/Triene Cycloaddition (provided in "dataset" folder)
<p align="center">
<img src="https://github.com/rshin1209/bgan_eps/assets/25111091/e78b318a-37d5-40ee-a6d3-b747457b03f3", width=50%>
</p>

The diene/triene cycloaddition is an ambimodal pericyclic reaction involving butadiene with hexatriene. It yields two products with asynchronous bond formations: 4+2-adduct (bond 1 and bond 2) and 6+4-adduct (bond 1 and bond 3)

<p align="center">
<img src="https://github.com/rshin1209/bgan_eps/assets/25111091/45e297e2-09dc-403d-908d-0f97f43d66bb", width=50%>
</p>

### The overview of BGAN-EPS

<p align="center">
<img src="https://github.com/rshin1209/bgan_eps/assets/25111091/c1b2280b-3ce6-4437-8699-7db437239b6b" width=100%>
</p>

#### Step 1: Quasiclassical Trajectory Simulation
        Functional/Basis Set: B3LYP-D3/6-31G(d)
        Integration Time Step: 1 fs
        Temperature: 298.15 K

Files to prepare:
1. Collect post-transition-state (post-TS) trajectories and combine them into a single xyz file (all post-TS trajectories) (e.g., ./dataset/dta_r2p_1.xyz).
2. Optimized TS structure file in pdb format (e.g., ./dataset/dta_r2p_TS.pdb).

**Filename format must be \[name of reaction\]\_r2p\_#.XXX**

#### Step 2: BGAN-assisted Configuration Sampling
##### Step 2.1: Coordinate Conversion

xyz2bat.py converts Cartesian coordinates of snapshots into redundant internal coordinates based on bonding connectivity (e.g., ./log/dta_r2p_1/topology.txt). The resulting internal coordinates are saved in a 2D numpy array (e.g., ./log/dta_r2p_1/dof.npy) with rows of snapshots and columns of internal coordinates.

        python xyz2bat.py --nb1 1 --nb2 10 --ts dta_r2p_TS --atom1 11 --atom2 13 --reaction dta_r2p_1
        python xyz2bat.py --nb1 1 --nb2 10 --ts dta_r2p_TS --atom1 2 --atom2 5 --reaction dta_r2p_2
        [nb1] -- first atom number in bond 1
        [nb2] -- second atom number in bond 1
        [atom1] -- first atom number in reaction coordinate (e.g., bond 2 or bond 3)
        [atom2] -- second atom number in reaction coordinate (e.g., bond 2 or bond 3)
        [reaction] -- Name of the reaction file without format tag
        [ts] -- Name of the optimized transition state structure file (pdb) without format tag

##### Step 2.2: BGAN Training and entropic path sampling (EPS)

main.py trains the BGAN model with internal coordinates of snapshots and performs entropic path sampling. The BGAN-EPS method runs for \[loop\] number of rounds to minimize statistical errors aroused by the deep generative model. The resulting entropy calculations are stored in (for example) './log/dta_r2p_1/bgan#, where # refers to round.

        python main.py --reaction dta_r2p_1 --bondmax 2.790 --bondmin 1.602 --ensemble 9
        python main.py --reaction dta_r2p_2 --bondmax 3.009 --bondmin 1.689 --ensemble 10
        [reaction] -- Name of the reaction file without format tag
        [ensemble] -- Number of structural ensembles for entropic path sampling
        [bondmax] -- Maximum bond length (i.e., bond length in the optimized TS structure)
        [bondmin] -- Minimum bond length (i.e., bond formation criterion)
        [temperature] -- Temperature in configurational entropy calculation
        [eps_type] -- Type of entropic path sampling: average or max (average recommended)

        [batch_size] -- Batch size for BGAN training (64 recommended)
        [epochs] -- Number of epochs for BGAN training (50 recommended)
        [lr] -- Learning rate of Adam Optimizer in BGAN training (1e-4 recommended)
        [beta1] -- Momentum1 for Adam Optimizer (0.5 recommended) 
        [beta2] -- Momentum2 for Adam Optimizer (0.999 recommended)
        [loop] -- Number of BGAN-EPS rounds (5-20 recommended based on available computation resources)

#### Step 3: Entropy Analysis
##### Step 3.1: Entropy Profiling

## Contact
Please open an issue in Github or contact wook.shin@vanderbilt.edu if you have any problem in BGAN-EPS.

## Citation
1. Shin, W.; Ran, X.; Yang, Z. J. Accelerated Entropic Path Sampling with a Bidirectional Generative Adversarial Network. The Journal of Physical Chemistry B 2023, 127 (19), 4254-4260. DOI: 10.1021/acs.jpcb.3c01202.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
