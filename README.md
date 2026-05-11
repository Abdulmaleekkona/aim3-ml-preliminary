# aim3-ml-preliminary
# Aim 3 — ML Preliminary Analysis: HFILOH Bonded Fraction Prediction

## Overview
This repository contains the preliminary machine learning analysis 
for Aim 3 of my PhD dissertation. As a proof of concept, I demonstrate 
that ML models can predict the bonded fraction of a nanometer-thick 
ionic liquid (HFILOH) coating from experimental conditions (concentration 
and dwell time).

## Background
HFILOH is a highly fluorinated imidazolium-based ionic liquid that forms 
a dual-layer structure (bonded + mobile layer) when dip-coated on silica. 
The bonded fraction — the proportion of irreversibly adsorbed IL — is a 
key predictor of tribological performance (CoF) and self-healing capability.

Data source: Dahiru & Li (2025) Langmuir — systematic concentration 
and dwell time measurements on silica substrates.

## Results
| Model | R² | MAE |
|---|---|---|
| Random Forest | 0.853 | 0.031 |
| Gradient Boosting | 0.858 | 0.031 |
| Gaussian Process | -0.051 | 0.082 |

SHAP analysis confirms the model learned physically correct relationships:
- Dwell time is the dominant driver of bonded fraction
- Higher concentration decreases bonded fraction (mobile layer grows faster)

## Repository Structure


## How to Run
```bash
# Clone the repo
git clone https://github.com/Abdulmaleekkona/aim3-ml-preliminary.git
cd aim3-ml-preliminary

# Create conda environment
conda create -n aim3-ml python=3.9
conda activate aim3-ml
pip install -r requirements.txt

# Open notebook
jupyter notebook notebooks/01_hfiloh_bonded_fraction_demo.ipynb
```

## Next Steps
- Add WCA and HCA targets once data is filled in
- Expand to full multi-IL dataset (~75 ILs) for Aim 3
- Add RDKit molecular descriptors from SMILES
- Virtual screening of candidate IL structures

## Author
Abdulmalik — PhD Candidate, University of Pittsburgh  