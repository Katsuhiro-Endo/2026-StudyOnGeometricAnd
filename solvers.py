from __future__ import annotations
import csv
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path
import numpy as np
from mechanics import PROBLEM_SPECS, load_model, levels_to_q, levels_to_free_u, batch_objective_log, add_domain_wall_penalty, objective_log, q_to_free_u, qubo_value, residual, free_dof_indices, constrained_dof_indices

def index_to_levels(indices: np.ndarray, base: int, dim: int) -> np.ndarray:
    levels = np.empty((len(indices), dim), dtype=np.int16)
    rem = indices.astype(np.int64, copy=True)
    for j in range(dim - 1, -1, -1):
        levels[:, j] = rem % base
        rem //= base
    return levels

def run_grid_search(model: dict, spec, chunk_size: int) -> dict:
    free_dim = int(np.sum(model['BC'] == 0))
    base = spec.num_bd + 1
    total = base ** free_dim
    best_value = float('inf')
    best_levels = None
    best_free_u = None
    start = time.perf_counter()
    for offset in range(0, total, chunk_size):
        stop = min(offset + chunk_size, total)
        indices = np.arange(offset, stop, dtype=np.int64)
        levels = index_to_levels(indices, base=base, dim=free_dim)
        free_us = levels_to_free_u(levels, num_bd=spec.num_bd, ran=spec.ran, shift=spec.free_shift)
        values = batch_objective_log(model, free_us)
        local = int(np.argmin(values))
        if float(values[local]) < best_value:
            best_value = float(values[local])
            best_levels = levels[local].astype(int).tolist()
            best_free_u = free_us[local].astype(float).tolist()
    elapsed = time.perf_counter() - start
    return {'method': 'objective_grid_search', 'objective_log10_r2_plus_1': best_value, 'levels': best_levels, 'free_u': best_free_u, 'total_combinations': total, 'chunk_size': chunk_size, 'elapsed_seconds': elapsed}

def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    csv_path = path.with_suffix('.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

def batch_qubo_value(q_flat: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    return np.einsum('bi,ij,bj->b', q_flat, q_matrix, q_flat, optimize=True)

def run_bruteforce(model: dict, spec, qij: np.ndarray, bias: float, chunk_size: int, lambda_dw: float) -> dict:
    free_dim = int(np.sum(model['BC'] == 0))
    base = spec.num_bd + 1
    total = base ** free_dim
    qij_dw = add_domain_wall_penalty(qij, free_dim=free_dim, num_bd=spec.num_bd, lambda_dw=lambda_dw)
    best_h = float('inf')
    best_levels = None
    best_q = None
    start = time.perf_counter()
    for offset in range(0, total, chunk_size):
        stop = min(offset + chunk_size, total)
        indices = np.arange(offset, stop, dtype=np.int64)
        levels = index_to_levels(indices, base=base, dim=free_dim)
        q = levels_to_q(levels, spec.num_bd).reshape(len(levels), free_dim * spec.num_bd)
        values = batch_qubo_value(q, qij_dw)
        local = int(np.argmin(values))
        if float(values[local]) < best_h:
            best_h = float(values[local])
            best_levels = levels[local].astype(int).tolist()
            best_q = q[local].astype(int).tolist()
    elapsed = time.perf_counter() - start
    free_u = levels_to_free_u(np.asarray(best_levels), spec.num_bd, spec.ran, spec.free_shift).astype(float)
    objective = float(batch_objective_log(model, free_u.reshape(1, -1))[0])
    return {'method': 'qubo_bruteforce', 'qubo_hamiltonian': best_h, 'qubo_hamiltonian_with_bias': best_h + float(np.asarray(bias).ravel()[0]), 'objective_log10_r2_plus_1': objective, 'levels': best_levels, 'free_u': free_u.tolist(), 'q_flat': best_q, 'total_combinations': total, 'chunk_size': chunk_size, 'lambda_dw': lambda_dw, 'elapsed_seconds': elapsed}

def diagnose_one(model: dict, free_u: list[float], label: str) -> tuple[dict, list[dict], list[dict]]:
    r, u, elements = residual(model, np.asarray(free_u, dtype=float))
    free = free_dof_indices(model)
    constrained = constrained_dof_indices(model)
    r_free = r[free]
    r2 = float(np.dot(r_free, r_free))
    summary = {'label': label, 'free_u': json.dumps(list(map(float, free_u))), 'objective_log10_r2_plus_1': float(np.log10(r2 + 1.0)), 'free_residual_norm': float(np.sqrt(r2)), 'free_residual_max_abs': float(np.max(np.abs(r_free))) if len(r_free) else 0.0, 'full_u': json.dumps(list(map(float, u)))}
    reactions = [{'label': label, 'dof_0based': dof, 'node': dof // 2 + 1, 'component': 'x' if dof % 2 == 0 else 'y', 'reaction': float(r[dof])} for dof in constrained]
    element_rows = []
    for row in elements:
        out = {'label': label}
        out.update(row)
        element_rows.append(out)
    return (summary, reactions, element_rows)

def parse_float_from_stem(stem: str) -> float | None:
    marker = '_fsrfm_std_'
    if marker not in stem:
        return None
    rest = stem.split(marker, 1)[1]
    value = rest.split('_', 1)[0]
    try:
        return float(value)
    except ValueError:
        return None

def domain_wall_violations(q_flat: np.ndarray, free_dim: int, num_bd: int) -> dict:
    q = np.asarray(q_flat, dtype=int).reshape(free_dim, num_bd)
    transitions = q[:, 1:] - q[:, :-1]
    violation_count = int(np.sum(transitions > 0))
    valid_blocks = np.sum(np.all(transitions <= 0, axis=1))
    levels = np.sum(q, axis=1).astype(int)
    return {'dw_violation_count': violation_count, 'dw_valid_blocks': int(valid_blocks), 'dw_total_blocks': int(free_dim), 'dw_all_valid': bool(violation_count == 0), 'levels_by_sum': levels.tolist()}

def build_amplify_model(q_matrix: np.ndarray):
    from amplify import Model, VariableGenerator
    q_matrix = np.asarray(q_matrix, dtype=float)
    n = q_matrix.shape[0]
    gen = VariableGenerator()
    q = gen.array('Binary', n)
    objective = 0.0
    for i in range(n):
        coeff = float(q_matrix[i, i])
        if coeff != 0.0:
            objective += coeff * q[i]
    for i in range(n):
        row = q_matrix[i]
        for j in range(i + 1, n):
            coeff = float(row[j] + q_matrix[j, i])
            if coeff != 0.0:
                objective += coeff * q[i] * q[j]
    return (Model(objective), q)

def make_client(timeout_ms: int, num_gpus: int):
    from amplify import FixstarsClient
    client = FixstarsClient()
    token = os.environ.get('AMPLIFY_TOKEN')
    url = os.environ.get('AMPLIFY_URL')
    if not token and (not url):
        raise ValueError('Set AMPLIFY_TOKEN or AMPLIFY_URL externally.')
    if token:
        client.token = token
    if url:
        client.url = url
    client.parameters.timeout = timedelta(milliseconds=timeout_ms)
    client.parameters.num_gpus = num_gpus
    client.parameters.outputs.num_outputs = 0
    return client

def solve_once(model, q, client) -> dict:
    from amplify import solve
    start = time.perf_counter()
    try:
        result = solve(model, client)
    except Exception:
        raise RuntimeError('Solver request failed; details suppressed to protect credentials.') from None
    elapsed = time.perf_counter() - start
    q_eval = np.asarray(q.evaluate(result.best.values), dtype=int).ravel()
    return {'solver_objective': float(result.best.objective), 'q_flat': q_eval, 'solve_elapsed_seconds': elapsed}

def summarize_solution(model_data: dict, spec, q_matrix_base: np.ndarray, q_matrix_dw: np.ndarray, bias: float, q_flat: np.ndarray, label: str) -> dict:
    free_dim = int(np.sum(model_data['BC'] == 0))
    dw = domain_wall_violations(q_flat, free_dim=free_dim, num_bd=spec.num_bd)
    free_u = q_to_free_u(q_flat, free_dim=free_dim, num_bd=spec.num_bd, ran=spec.ran, shift=spec.free_shift)
    summary, reactions, elements = diagnose_one(model_data, free_u.tolist(), label)
    yielded_count = int(sum((1 for row in elements if row.get('yielded'))))
    axial_abs_max = float(max((abs(float(row['axial_force'])) for row in elements), default=0.0))
    stress_abs_max = float(max((abs(float(row['stress'])) for row in elements), default=0.0))
    out = {'free_u': free_u.astype(float).tolist(), 'objective_log10_r2_plus_1': float(objective_log(model_data, free_u)), 'free_residual_norm': float(summary['free_residual_norm']), 'free_residual_max_abs': float(summary['free_residual_max_abs']), 'qubo_hamiltonian_base': float(qubo_value(q_flat, q_matrix_base, 0.0)), 'qubo_hamiltonian_dw': float(qubo_value(q_flat, q_matrix_dw, 0.0)), 'qubo_hamiltonian_dw_with_bias': float(qubo_value(q_flat, q_matrix_dw, bias)), 'yielded_element_count': yielded_count, 'axial_force_abs_max': axial_abs_max, 'stress_abs_max': stress_abs_max}
    out.update(dw)
    return out

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def csv_ready(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, (list, dict, tuple)):
            out[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        elif isinstance(value, float) and (not math.isfinite(value)):
            out[key] = str(value)
        else:
            out[key] = value
    return out

def solve_bruteforce(problem, checkpoint, output, lambda_dw):
    spec = PROBLEM_SPECS[problem]
    model = load_model(problem)
    with np.load(checkpoint, allow_pickle=False) as data:
        result = run_bruteforce(model, spec, data['Qij'], data['bias'], 100000, lambda_dw)
    result['problem'] = problem
    result['npz'] = str(checkpoint)
    write_result(output / (checkpoint.stem + '_qubo_bruteforce.json'), result)
    return result


def solve_amplify(problem, checkpoint, output, options):
    spec = PROBLEM_SPECS[problem]
    model_data = load_model(problem)
    with np.load(checkpoint, allow_pickle=False) as data:
        q_matrix_base = np.asarray(data['Qij'], dtype=float)
        bias = float(np.asarray(data['bias']).ravel()[0])
    free_dim = int(np.sum(model_data['BC'] == 0))
    lambda_dw = options['lambda_dw']
    q_matrix_dw = add_domain_wall_penalty(q_matrix_base, free_dim=free_dim, num_bd=spec.num_bd, lambda_dw=lambda_dw)
    build_start = time.perf_counter()
    amplify_model, q = build_amplify_model(q_matrix_dw)
    build_elapsed = time.perf_counter() - build_start
    client = make_client(options['timeout_ms'], options['num_gpus'])
    import amplify
    backend = getattr(amplify, '__version__', 'unknown')
    private_test = bool(getattr(solve_once, 'is_private_test', False))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'execution.json').write_text(json.dumps({'backend_version': backend,
        'is_private_test': private_test, 'checkpoint': str(checkpoint)}, indent=2))
    rows = []
    for repeat in range(options['calls_per_model']):
        solved = solve_once(amplify_model, q, client)
        label = ('local_sa_validation' if private_test else 'reproduction') + f'_repeat_{repeat}'
        summary = summarize_solution(model_data, spec, q_matrix_base, q_matrix_dw, bias, solved['q_flat'], label)
        row = {'problem': problem, 'tag': label, 'repeat': repeat, 'npz': str(checkpoint),
            'npz_stem': checkpoint.stem, 'lambda_sr': parse_float_from_stem(checkpoint.stem),
            'lambda_dw': lambda_dw, 'timeout_ms': options['timeout_ms'], 'num_gpus': options['num_gpus'],
            'amplify_sdk_version': backend, 'amplify_solver_version': None if private_test else getattr(client, 'version', None),
            'model_build_elapsed_seconds': build_elapsed, 'solver_objective': solved['solver_objective'],
            'solve_elapsed_seconds': solved['solve_elapsed_seconds'], 'q_flat': solved['q_flat'].astype(int).tolist()}
        row.update(summary)
        rows.append(row)
        print(json.dumps(csv_ready(row)), flush=True)
    with (output / 'solver_results.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(csv_ready(row)) + '\n')
    write_csv(output / 'solver_results.csv', [csv_ready(row) for row in rows])
    return rows
