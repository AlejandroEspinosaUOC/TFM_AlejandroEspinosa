# TFM_AlejandroEspinosa
Repository containing the code for the Master Thesis of the student Alejandro Espinosa Polo.

**Abstract of the thesis:**
The growing interest in creating applications that can run closer to the user has driven the rise of edge-cloud continuum environments, which provide close computation while maintaining security, privacy and cost-efficiency. These advancements are allowing the creation of new use cases where privacy, execution time or even battery life of the device may be crucial in their existence. These environments, however, present significant challenges in the management of their resources due to their heterogeneity, this is why standardization solutions like Kubernetes (K8s) are used, which simplifies the process of application deployment. In this thesis we leverage the power of Multi Agent Reinforcement Learning (MARL) to solve the challenging optimisation problem of application placement, with the focus on energy optimisation.

This thesis aims to expand current research of the student, focusing on a single agent Deep Reinforcement Learning (DRL) that optimizes energy consumption of the applications placed in the system. This expansion will create an auction based MARL that will optimise energy consumption of all applications that are executed in the system. For this purpose, we will create an edge-cloud emulator where we will train and validate our MARL model. 

## Experiments 
Experiments done during the thesis can be found in folders results_base_marl_exp and results_tfm.
If the user wants to test the system by themselves, an explanaition follows.

## How to use

Before going all the functionalities of the code generated for the thesis, we encourage the user to create a virtual envrionment for the required dependencies:

Python >= 3.10.12 is required, then create the virtual env:

```
python3 -m venv .venv
source .venv/bin/activate 
```

Once the venv is created, install all requirements:

```
pip install -r requirements.txt
```

## Model Training
Model Training has been configured to be compatible both with cpu and gpu arquitectures. You can train the following models:
- Base models with drl_agents_train_main.py
- Complex Betting MARL agents with complex_marl_agents_training.py


In order to launch the training script for the Complex Betting MARL it is needed to call the script like this, where X is the numner of agents to train:
```
python complex_marl_agents_training.py --num-agents X --enable-new-api-stack
```

## Data Generation
If needed (altough data is provided in data folder), data can be generated with the following scripts:
- data_gen.py: Application data to train base models
- node_data_gen.py: node arquitectures
- sythetic_data_tfm.py: Bet generation for complex MARL training


# Acknowledgments
This work has received funding from the European Commission Horizon Europe programme under grant agreement number 101092696 (CODECO) and from the Swiss State Secretariat for Education, Research and Innovation (SERI) under contract number 23.00028.
