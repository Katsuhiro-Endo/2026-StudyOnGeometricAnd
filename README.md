# Reproduction Code for "Study on Geometric and Materially Nonlinear Truss Structure Analysis Using Ising Machine and Factorization Machine"

```sh
python -m pip install -r requirements.txt
```

Run from this directory. Output directories must be new and outside this package.

| Task | Command |
|---|---|
| Train FM | `python reproduce.py train problem_1 ../p1_training` |
| Train FM and solve with Amplify | `python reproduce.py pipeline problem_1 ../p1_pipeline` |
| Grid search of the structural objective | `python reproduce.py grid problem_1 ../p1_grid` |
| Brute-force search of a trained QUBO | `python reproduce.py qubo problem_1 ../p1_qubo` |
| Solve a trained QUBO with Amplify | `python reproduce.py amplify problem_1 ../p1_amplify` |
| All NR histories | `python reproduce.py nr problem_1 ../nr_all_problems` |
| All 115 seed/lambda fits | `python reproduce.py sweep problem_1 ../lambda_seed_sweep` |

For the first five commands, replace `problem_1` with `problem_2` or `problem_3` and choose a matching output directory. `nr` and `sweep` always run all problems. Without arguments, `python reproduce.py` trains Problem 1 only.

Set `AMPLIFY_TOKEN` externally before `pipeline` or `amplify` (paid service).

For the usual train-and-solve workflow, use `pipeline`. It trains the model and solves the resulting QUBO automatically; no `run.checkpoint` setting is needed.

Settings are in `experiment_config.json`. For separate steps:
- `qubo` and `amplify` do not train a model. First run `train`, then set `run.checkpoint` to the generated NPZ file for that problem.
- To train just one regularization-study model, set `run.training_family` to `"sweep"` and use `train`. The `sweep` command instead trains all 115 conditions.

All code has been organized or developed with the assistance of AI.