import sys
sys.dont_write_bytecode = True
import csv
import json
import os
from pathlib import Path
import mechanics
import training
import solvers
import nr


def settings():
    return json.loads((Path(__file__).parent / 'experiment_config.json').read_text())


def train_selected(problem, output, options):
    selected = options['selected_models'][problem]
    return training.train(problem, output, lambda_sr=selected['lambda_sr'],
        train_seed=selected['train_seed'], epochs=options['run']['epochs'],
        family=options['run']['training_family'])


def sweep(output, options):
    rows = []
    for problem, seeds in options['sweep_seeds'].items():
        spec = mechanics.PROBLEM_SPECS[problem]
        model = mechanics.load_model(problem)
        for seed in seeds:
            for lam in spec.lambda_values:
                folder = output / problem / f'seed_{seed}' / f'lambda_{lam:g}'
                training.train(problem, folder, lambda_sr=lam, train_seed=seed,
                               epochs=options['run']['epochs'], family='sweep')
                with next(folder.glob('*metrics.csv')).open() as f:
                    row = next(csv.DictReader(f))
                row['condition_dir'] = str(folder)
                if problem == 'problem_1':
                    row = training.refresh_problem1_row(row, model)
                rows.append(row)
                training.save_csv(output / 'lambda_seed_sweep_summary.csv', rows, list(rows[0]))
    return rows


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else 'train'
    problem = sys.argv[2] if len(sys.argv) > 2 else 'problem_1'
    options = settings()
    output = Path(sys.argv[3] if len(sys.argv) > 3 else '../run_' + action + '_' + problem).resolve()
    package = Path(__file__).resolve().parent
    if output == package or package in output.parents or output.exists():
        raise ValueError('Use a new output directory outside this package')
    if problem not in mechanics.PROBLEM_SPECS:
        raise ValueError('Unknown problem')
    if action not in ('train', 'pipeline', 'grid', 'qubo', 'amplify', 'nr', 'sweep'):
        raise ValueError('Choose train, pipeline, grid, qubo, amplify, nr, or sweep')
    if action in ('pipeline', 'amplify') and not (os.environ.get('AMPLIFY_TOKEN') or os.environ.get('AMPLIFY_URL')):
        raise ValueError('Set Amplify credentials externally before running')
    if action == 'train':
        train_selected(problem, output, options)
    elif action == 'pipeline':
        checkpoint = train_selected(problem, output / 'training', options)
        solvers.solve_amplify(problem, checkpoint, output / 'solving', options)
    elif action in ('amplify', 'qubo'):
        checkpoint = Path(options['run']['checkpoint']).resolve()
        if not checkpoint.is_file():
            raise ValueError('Set run.checkpoint to a newly generated NPZ file')
        if action == 'amplify':
            solvers.solve_amplify(problem, checkpoint, output, options)
        else:
            solvers.solve_bruteforce(problem, checkpoint, output, options['lambda_dw'])
    elif action == 'grid':
        model = mechanics.load_model(problem)
        result = solvers.run_grid_search(model, mechanics.PROBLEM_SPECS[problem], 200000)
        solvers.write_result(output / (problem + '_grid_search.json'), result)
    elif action == 'nr':
        nr.run(output, options)
    else:
        sweep(output, options)


if __name__ == '__main__':
    main()
