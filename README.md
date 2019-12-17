# CS330 Project: Clustered Meta Learning

This project aims to improve the performance of Model-Agnostic Meta Learning (MAML) by having multiple meta-parameters that enable the algorithm to understand the relations between various tasks.

To run the algorithm, select the desired configuration in `run_scripts/maml_run_mujoco.py` and run

```python run_scripts/maml_run_mujoco.py```

To load a model change the loading flag in `meta_policy_search/meta_trainer.py` and provide the checkpoint directory.


## Acknowledgements
This repository includes environments introduced in ([Duan et al., 2016](https://arxiv.org/abs/1611.02779), 
[Finn et al., 2017](https://arxiv.org/abs/1703.03400)).
