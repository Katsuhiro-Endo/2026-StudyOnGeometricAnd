from __future__ import annotations
import json
import math
import numpy as np
import mechanics as core
from mechanics import residual

def force_law(length, length0, model, mode, backend=math):
    e_mod, area = (model['E'], model['A'])
    stretch = length / length0
    if mode == 'log_current':
        return (e_mod * area * backend.log(stretch), e_mod * area / length)
    if mode == 'engineering_current':
        return (e_mod * area * (stretch - 1), e_mod * area / length0)
    if mode == 'green_lagrange_stvk':
        strain = (stretch * stretch - 1) / 2
        return (e_mod * area * strain * stretch, e_mod * area * (3 * stretch * stretch - 1) / (2 * length0))
    if mode == 'log_piola':
        log_stretch = backend.log(stretch)
        return (e_mod * area * log_stretch / stretch, e_mod * area * (1 - log_stretch) / (length0 * stretch * stretch))
    strain = stretch - 1
    epsy = model['sigmay'] / e_mod
    tangent = e_mod
    if abs(strain) <= epsy:
        stress = e_mod * strain
    else:
        tangent = e_mod * model['H'] / (e_mod + model['H'])
        stress = (1 if strain >= 0 else -1) * (model['sigmay'] + tangent * (abs(strain) - epsy))
    return (area * stress, area * tangent / length0)

def independent_residual_tangent(model, free_u, mode):
    xy = np.asarray(model['xy'], dtype=float)
    u = np.zeros_like(xy)
    bc = np.asarray(model['BC'])
    free = np.flatnonzero(bc.ravel() == 0)
    u.ravel()[free] = free_u
    for node, component, value in model.get('pre_d', []):
        u[node - 1, component] = value
    r = np.zeros(xy.size)
    tangent = np.zeros((xy.size, xy.size))
    for node_i, node_j in model['node_label']:
        i, j = (node_i - 1, node_j - 1)
        initial = xy[j] - xy[i]
        current = initial + u[j] - u[i]
        length0, length = (np.linalg.norm(initial), np.linalg.norm(current))
        if length < 1e-12:
            raise ValueError('collapsed truss element')
        n = current / length
        axial, derivative = force_law(length, length0, model, mode)
        nn = np.outer(n, n)
        k = derivative * nn + axial / length * (np.eye(2) - nn)
        ii, jj = (np.array([2 * i, 2 * i + 1]), np.array([2 * j, 2 * j + 1]))
        r[ii] -= axial * n
        r[jj] += axial * n
        tangent[np.ix_(ii, ii)] += k
        tangent[np.ix_(jj, jj)] += k
        tangent[np.ix_(ii, jj)] -= k
        tangent[np.ix_(jj, ii)] -= k
    for node, component, value in model.get('pre_f', []):
        r[2 * (node - 1) + component] -= value
    return (r[free], tangent[np.ix_(free, free)])

def newton(model, mode, initial, tag, line_search=False, max_steps=100):
    u = np.array(initial, dtype=float)
    scale = model['E'] * model['A']
    history = []
    status = 'max_iterations'
    for iteration in range(max_steps + 1):
        r, k = independent_residual_tangent(model, u, mode)
        norm = float(np.linalg.norm(r))
        history.append({'tag': tag, 'iteration': iteration, 'residual_norm_N': norm, 'relative_to_EA': norm / scale, 'u': u.tolist(), 'step_alpha': None})
        if norm / scale <= 1e-13:
            status = 'residual_converged'
            break
        if iteration == max_steps:
            break
        try:
            delta = np.linalg.solve(k / scale, -r / scale)
        except np.linalg.LinAlgError:
            status = 'singular_tangent'
            break
        alpha = 1.0
        if line_search:
            for _ in range(25):
                trial, _ = independent_residual_tangent(model, u + alpha * delta, mode)
                if np.linalg.norm(trial) < norm:
                    break
                alpha *= 0.5
        history[-1]['step_alpha'] = alpha
        u += alpha * delta
        if not np.all(np.isfinite(u)):
            status = 'nonfinite_iterate'
            break
    r, k = independent_residual_tangent(model, u, mode)
    row = {'tag': tag, 'mode': mode, 'initial_u': list(map(float, initial)), 'free_u': u.tolist(), 'status': status, 'iterations': len(history) - 1, 'residual_norm_N': float(np.linalg.norm(r)), 'relative_to_EA': float(np.linalg.norm(r) / scale), 'objective': float(np.log10(np.dot(r, r) + 1)), 'tangent_eigenvalues': np.linalg.eigvalsh(k).tolist()}
    return (row, history)

def cycle_info(history):
    us = np.array([row['u'] for row in history])
    if len(us) < 41:
        return None
    for p in range(1, 11):
        diff = us[-3 * p:] - us[-4 * p:-p]
        if np.max(np.linalg.norm(diff, axis=1) / np.maximum(1, np.linalg.norm(us[-3 * p:], axis=1))) < 1e-07:
            return {'period': p, 'last_cycle_u': us[-p:].tolist(), 'last_cycle_residual_norm_N': [row['residual_norm_N'] for row in history[-p:]]}
    return None

def scalar(initial, model, checked, eq, limit=20):
    scale = model['E'] * model['A']
    tol = 1e-13
    u = np.array(initial, float)
    history = []
    status = 'max_iterations'
    for step in range(limit + 1):
        with np.errstate(all='ignore'):
            try:
                r, k = independent_residual_tangent(model, u, 'log_current')
                norm = float(np.linalg.norm(r))
            except (ValueError, OverflowError, FloatingPointError):
                status = 'nonfinite'
                break
        if not np.isfinite(norm) or not np.all(np.isfinite(k)):
            status = 'nonfinite'
            break
        history.append(dict(iteration=step, u=u.tolist(), residual_norm_N=norm))
        if norm / scale <= tol:
            status = 'converged'
            break
        if step == limit:
            break
        try:
            u += np.linalg.solve(k / scale, -r / scale)
        except np.linalg.LinAlgError:
            status = 'singular'
            break
    result = dict(initial=list(map(float, initial)), status=status, updates=history[-1]['iteration'], cycle=cycle_info(history))
    h = np.array([p['u'] for p in history])
    result['max_abs_m'] = float(np.max(abs(h)))
    if len(history) > 20:
        common, _, _ = core.residual(checked, h[20])
        norm = float(np.linalg.norm(common[core.free_dof_indices(checked)]))
        assert np.isclose(norm, history[20]['residual_norm_N'], rtol=1e-10, atol=0.001)
        distance = float(np.linalg.norm(h[20] - eq))
        result.update(u20=h[20].tolist(), residual20_N=norm, objective20=float(np.log10(norm * norm + 1)), distance20_m=distance, genuinely_nonconverged20=bool(norm / scale > tol), max_abs_0_20_m=float(np.max(abs(h[:21]))))
        dr = np.diff(h[:21], axis=0)
        result['reversals_0_20'] = int(np.sum(np.any(dr[1:] * dr[:-1] < 0, axis=1)))
        result['residual_ups_0_20'] = int(np.sum(np.diff([p['residual_norm_N'] for p in history[:21]]) > 0))
    return (result, history)

def scalar_res(u: float, model) -> float:
    return float(residual(model, [u])[0][3])

def nr_history(u0: float, model, max_iter: int=30) -> list[tuple[int, float, float]]:
    out = []
    u = float(u0)
    for i in range(max_iter + 1):
        r = scalar_res(u, model)
        out.append((i, u, r))
        if abs(r) < 1e-06:
            break
        h = 1e-05 * max(1.0, abs(u))
        dr = (scalar_res(u + h, model) - scalar_res(u - h, model)) / (2.0 * h)
        if dr == 0.0 or not np.isfinite(dr):
            break
        u = u - r / dr
        if abs(u) > 1000000.0:
            break
    return out

def run(output, settings):
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    p1model = core.load_model('problem_1')
    for case in settings['nr']['problem_1']:
        initial = case['initial_u']
        history = nr_history(initial, p1model, max_iter=case['maximum_updates'])
        key = f'problem_1_initial_{initial:g}'
        (output / (key + '_history.json')).write_text(json.dumps(history, indent=2))
        results[key] = {'updates': len(history) - 1, 'free_u': [history[-1][1]],
                        'residual_norm_N': abs(history[-1][2])}
    for problem, cases in settings['nr']['multivariable'].items():
        spec = core.PROBLEM_SPECS[problem]
        raw = settings['models'][spec.model_json.as_posix()]
        model = core.load_model(problem)
        converged_u = None
        for name in ['converged_reference', 'nonconverged_example']:
            case = cases[name]
            if problem == 'problem_2' and name == 'nonconverged_example':
                _, history = scalar(case['initial_u'], raw, model, np.asarray(converged_u),
                    limit=settings['nr']['maximum_updates'])
                u = history[-1]['u']
            else:
                result, history = newton(raw, case['mode'], case['initial_u'],
                    problem + '_' + name, max_steps=settings['nr']['maximum_updates'])
                u = result['free_u']
                if name == 'converged_reference':
                    converged_u = u
            residual_value, _, _ = core.residual(model, np.asarray(u))
            norm = float(np.linalg.norm(residual_value[core.free_dof_indices(model)]))
            key = problem + '_' + name
            (output / (key + '_history.json')).write_text(json.dumps(history, indent=2))
            results[key] = {'free_u': list(u), 'updates': history[-1]['iteration'],
                'residual_norm_N': norm, 'converged': norm / (model['E'] * model['A']) <= 1e-13,
                'objective': float(np.log10(norm * norm + 1))}
    (output / 'nr_results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    return results
