# TutorTest
This repository contains the code accompanying the paper:
“TutorTest: Evaluating Language Model-based Tutoring Policies Using Surrogate Tasks”

Venue: NeurIPS SEA Workshop 2025

Authors: Aishwarya Mandyam, Omer Gottesman, Sohrab Andaz, Dean Foster

# Overview
TutorTest provides a framework to: (1) Prepare and preprocess offline tutoring datasets, (2) Run experiments on CIMA and Khan Academy and (3) Compute baseline metrics for comparison

# Requirements
Python 3.9

# Datasets
Place all raw data under offline_data/. 
## CIMA
Download: https://aclanthology.org/2020.bea-1.5/
Put files under: offline_data/cima/

## Khan Academy
Access details and dataset info: https://blog.khanacademy.org/introducing-a-new-dataset-to-further-the-field-of-ai-research/
Put files under: offline_data/khan_academy/

# Preprocessing
Preprocess each dataset before running experiments.
```python
python src/preprocess_cima_dataset.py 
python src/preprocess_khan_academy_dataset.py
```

# Running Experiments
```python
python experiments/run_tutortest_cima.py
python experiments/run+tutorttest_khan_academy.py
```

# Contact
For questions or issues:

Contact: Aishwarya Mandyam (am2@stanford.edu)

