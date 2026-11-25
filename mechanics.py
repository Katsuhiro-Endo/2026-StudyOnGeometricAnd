from __future__ import annotations
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class ProblemSpec:
    name: str
    input_name: str
    model_json: Path
    ran: float
    num_bd: int
    num_dataset: int
    learning_rate: float
    default_epochs: int
    lambda_values: tuple[float, ...]
    free_shift: float = 0.0
PROBLEM_SPECS = {'problem_1': ProblemSpec(name='problem_1', input_name='8geo', model_json=Path('problem_1/Itruss8geo.json'), ran=25.0, num_bd=50, num_dataset=10, learning_rate=0.01, default_epochs=30000, lambda_values=(0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0), free_shift=-15.0), 'problem_2': ProblemSpec(name='problem_2', input_name='7geo', model_json=Path('problem_2/Itruss7geo.json'), ran=3.0, num_bd=60, num_dataset=100, learning_rate=0.01, default_epochs=300000, lambda_values=(0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0)), 'problem_3': ProblemSpec(name='problem_3', input_name='7mat', model_json=Path('problem_3/Itruss7mat.json'), ran=3.0, num_bd=60, num_dataset=100, learning_rate=0.01, default_epochs=300000, lambda_values=(0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0))}

def free_dof_indices(model: dict) -> list[int]:
    return [2 * node + comp for node in range(model['BC'].shape[0]) for comp in range(2) if model['BC'][node, comp] == 0]

def constrained_dof_indices(model: dict) -> list[int]:
    return [2 * node + comp for node in range(model['BC'].shape[0]) for comp in range(2) if model['BC'][node, comp] != 0]

def external_force(model: dict) -> np.ndarray:
    f_ext = np.zeros(model['xy'].shape[0] * 2, dtype=float)
    for node, comp, value in model['pre_f']:
        f_ext[2 * (node - 1) + comp] = value
    return f_ext

def build_full_displacement(model: dict, free_u: np.ndarray) -> np.ndarray:
    free_u = np.asarray(free_u, dtype=float).ravel()
    expected = len(free_dof_indices(model))
    if len(free_u) != expected:
        raise ValueError(f'expected {expected} free displacement values, got {len(free_u)}')
    u = np.zeros(model['xy'].shape[0] * 2, dtype=float)
    pos = 0
    for node in range(model['BC'].shape[0]):
        for comp in range(2):
            dof = 2 * node + comp
            bc = model['BC'][node, comp]
            if bc == 0:
                u[dof] = free_u[pos]
                pos += 1
            elif bc == -1:
                for pre_node, pre_comp, value in model['pre_d']:
                    if node + 1 == pre_node and comp == pre_comp:
                        u[dof] = value
                        break
    return u

def element_response(model: dict, u: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    xy = model['xy']
    e_mod = float(model['E'])
    area = float(model['A'])
    has_hardening = 'sigmay' in model and 'H' in model
    sigmay = float(model.get('sigmay', 0.0))
    hardening = float(model.get('H', 0.0))
    epsy = sigmay / e_mod if has_hardening else None
    f_int = np.zeros(xy.shape[0] * 2, dtype=float)
    elements = []
    for idx, (n1, n2) in enumerate(model['node_label'], start=1):
        i = int(n1) - 1
        j = int(n2) - 1
        dx0 = xy[j, 0] - xy[i, 0]
        dy0 = xy[j, 1] - xy[i, 1]
        dux = u[2 * j] - u[2 * i]
        duy = u[2 * j + 1] - u[2 * i + 1]
        dx = dx0 + dux
        dy = dy0 + duy
        length0 = math.hypot(dx0, dy0)
        length = math.hypot(dx, dy)
        cos = dx / length
        sin = dy / length
        if has_hardening:
            strain = length / length0 - 1.0
            yielded = abs(strain) > epsy
            if yielded:
                sign = 1.0 if strain >= 0.0 else -1.0
                stress = sign * (sigmay + e_mod * hardening / (e_mod + hardening) * (abs(strain) - epsy))
            else:
                stress = e_mod * strain
        else:
            strain = math.log(length / length0)
            stress = e_mod * strain
            yielded = False
        axial_force = area * stress
        local = np.array([-axial_force * cos, -axial_force * sin, axial_force * cos, axial_force * sin])
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        for dof, value in zip(dofs, local):
            f_int[dof] += value
        elements.append({'element': idx, 'node_i': int(n1), 'node_j': int(n2), 'length0': length0, 'length': length, 'strain': strain, 'stress': stress, 'axial_force': axial_force, 'yielded': yielded})
    return (f_int, elements)

def residual(model: dict, free_u: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    u = build_full_displacement(model, free_u)
    f_int, elements = element_response(model, u)
    return (f_int - external_force(model), u, elements)

def objective_r2(model: dict, free_u: np.ndarray) -> float:
    r, _, _ = residual(model, free_u)
    free = free_dof_indices(model)
    return float(np.dot(r[free], r[free]))

def objective_log(model: dict, free_u: np.ndarray) -> float:
    return math.log10(objective_r2(model, free_u) + 1.0)

def batch_objective_log(model: dict, free_us: np.ndarray) -> np.ndarray:
    free_us = np.asarray(free_us, dtype=float)
    if free_us.ndim == 1:
        free_us = free_us.reshape(1, -1)
    return np.asarray([objective_log(model, u) for u in free_us], dtype=float)

def dw_encode(level: int, num_bd: int) -> np.ndarray:
    q = np.zeros(num_bd, dtype=np.float32)
    q[:level] = 1.0
    return q

def dw_decode(q: np.ndarray, ran: float, shift: float=0.0) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    num_bd = q.shape[1]
    values = np.zeros(q.shape[0], dtype=float)
    for row in range(q.shape[0]):
        level = 0.0
        for i in range(num_bd - 1):
            level += (i + 1) * (q[row, i] - 2 * q[row, i] * q[row, i + 1] + q[row, i + 1])
        level += num_bd * q[row, num_bd - 1]
        values[row] = ran * (2.0 * (level / num_bd) - 1.0) + shift
    return values

def q_to_free_u(q_flat: np.ndarray, free_dim: int, num_bd: int, ran: float, shift: float=0.0) -> np.ndarray:
    q = np.asarray(q_flat, dtype=float).reshape(free_dim, num_bd)
    return dw_decode(q, ran=ran, shift=shift)

def levels_to_q(levels: np.ndarray, num_bd: int) -> np.ndarray:
    levels = np.asarray(levels, dtype=int)
    rows = [dw_encode(int(level), num_bd) for level in levels.ravel()]
    return np.asarray(rows, dtype=np.float32).reshape(*levels.shape, num_bd)

def levels_to_free_u(levels: np.ndarray, num_bd: int, ran: float, shift: float=0.0) -> np.ndarray:
    levels = np.asarray(levels, dtype=float)
    return ran * (2.0 * levels / num_bd - 1.0) + shift

def generate_unique_indices(start: int, end: int, count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if count > end - start + 1:
        raise ValueError('count is larger than the available range')
    out = set()
    while len(out) < count:
        out.add(rng.randint(start, end))
    return list(out)

def generate_dataset_levels(free_dim: int, num_bd: int, count: int, seed: int, include_zero: bool) -> np.ndarray:
    indices = generate_unique_indices(0, num_bd ** free_dim - 1, count, seed)
    random.Random(seed + 1000003).shuffle(indices)
    levels = np.zeros((count, free_dim), dtype=int)
    for i, index in enumerate(indices):
        rem = index
        for j in range(free_dim):
            base = num_bd ** (free_dim - j - 1)
            levels[i, j] = rem // base
            rem -= levels[i, j] * base
    if include_zero:
        zero = np.where(np.all(levels == 0, axis=1))[0]
        if len(zero):
            levels[[0, zero[0]]] = levels[[zero[0], 0]]
        else:
            levels[0, :] = 0
    return levels

def make_train_validation_solver(free_dim: int, num_bd: int, num_train: int, num_validation: int, train_seed: int, validation_seed: int) -> tuple[np.ndarray, np.ndarray]:
    train = generate_dataset_levels(free_dim, num_bd, num_train, train_seed, include_zero=True)
    validation = generate_dataset_levels(free_dim, num_bd, num_validation, validation_seed, include_zero=False)
    train_set = {tuple(row) for row in train}
    validation = np.asarray([row for row in validation if tuple(row) not in train_set], dtype=int)
    return (train, validation)

def qubo_value(q_flat: np.ndarray, q_matrix: np.ndarray, bias: float=0.0) -> float:
    q_flat = np.asarray(q_flat, dtype=float).ravel()
    return float(q_flat @ q_matrix @ q_flat + bias)

def add_domain_wall_penalty(q_matrix: np.ndarray, free_dim: int, num_bd: int, lambda_dw: float=15.0) -> np.ndarray:
    q = np.array(q_matrix, dtype=float, copy=True)
    for block in range(free_dim):
        offset = block * num_bd
        for local in range(num_bd):
            i = offset + local
            if local != 0:
                q[i, i] += lambda_dw
            if local + 1 < num_bd:
                q[i, i + 1] += -lambda_dw
    return q

def make_train_validation_sweep(free_dim: int, num_bd: int, num_train: int, num_validation: int, train_seed: int, validation_seed: int) -> tuple[np.ndarray, np.ndarray]:
    train = generate_dataset_levels(free_dim, num_bd, num_train, train_seed, include_zero=True)
    train_set = {tuple(row) for row in train}
    total_count = num_bd ** free_dim
    available_count = total_count - len(train_set)
    if num_validation >= available_count:
        validation_rows = []
        for index in range(total_count):
            rem = index
            row = []
            for j in range(free_dim):
                base = num_bd ** (free_dim - j - 1)
                level = rem // base
                rem -= level * base
                row.append(level)
            if tuple(row) not in train_set:
                validation_rows.append(row)
        validation = np.asarray(validation_rows, dtype=int)
    else:
        validation = generate_dataset_levels(free_dim, num_bd, num_validation, validation_seed, include_zero=False)
        validation = np.asarray([row for row in validation if tuple(row) not in train_set], dtype=int)
    return (train, validation)

def batch_full_displacement(model: dict, free_us: np.ndarray) -> np.ndarray:
    free_us = np.asarray(free_us, dtype=float)
    if free_us.ndim == 1:
        free_us = free_us.reshape(1, -1)
    sample = build_full_displacement(model, free_us[0])
    out = np.zeros((free_us.shape[0], sample.size), dtype=float)
    free = free_dof_indices(model)
    for dof in range(sample.size):
        out[:, dof] = sample[dof]
    for col, dof in enumerate(free):
        out[:, dof] = free_us[:, col]
    return out

def batch_objective_r2(model: dict, free_us: np.ndarray) -> np.ndarray:
    free_us = np.asarray(free_us, dtype=float)
    if free_us.ndim == 1:
        free_us = free_us.reshape(1, -1)
    u = batch_full_displacement(model, free_us)
    xy = model['xy']
    e_mod = float(model['E'])
    area = float(model['A'])
    has_hardening = 'sigmay' in model and 'H' in model
    sigmay = float(model.get('sigmay', 0.0))
    hardening = float(model.get('H', 0.0))
    epsy = sigmay / e_mod if has_hardening else None
    f_int = np.zeros((u.shape[0], xy.shape[0] * 2), dtype=float)
    for n1, n2 in model['node_label']:
        i = int(n1) - 1
        j = int(n2) - 1
        dx0 = xy[j, 0] - xy[i, 0]
        dy0 = xy[j, 1] - xy[i, 1]
        dux = u[:, 2 * j] - u[:, 2 * i]
        duy = u[:, 2 * j + 1] - u[:, 2 * i + 1]
        dx = dx0 + dux
        dy = dy0 + duy
        length0 = math.hypot(dx0, dy0)
        length = np.sqrt(dx * dx + dy * dy)
        cos = dx / length
        sin = dy / length
        if has_hardening:
            strain = length / length0 - 1.0
            abs_strain = np.abs(strain)
            elastic = abs_strain <= epsy
            stress = np.empty_like(strain)
            stress[elastic] = e_mod * strain[elastic]
            sign = np.where(strain >= 0.0, 1.0, -1.0)
            stress[~elastic] = sign[~elastic] * (sigmay + e_mod * hardening / (e_mod + hardening) * (abs_strain[~elastic] - epsy))
        else:
            strain = np.log(length / length0)
            stress = e_mod * strain
        axial = area * stress
        f_int[:, 2 * i] += -axial * cos
        f_int[:, 2 * i + 1] += -axial * sin
        f_int[:, 2 * j] += axial * cos
        f_int[:, 2 * j + 1] += axial * sin
    residual = f_int - external_force(model).reshape(1, -1)
    free = free_dof_indices(model)
    return np.sum(residual[:, free] * residual[:, free], axis=1)

def batch_objective_log(model: dict, free_us: np.ndarray) -> np.ndarray:
    return np.log10(batch_objective_r2(model, free_us) + 1.0)

def load_model(problem):
    settings = json.loads((Path(__file__).parent / 'experiment_config.json').read_text())
    data = settings['models'][PROBLEM_SPECS[problem].model_json.as_posix()]
    data['xy'] = np.asarray(data['xy'], dtype=float)
    data['BC'] = np.asarray(data['BC'], dtype=int)
    data['node_label'] = np.asarray(data['node_label'], dtype=int)
    data['pre_d'] = data.get('pre_d', [])
    data['pre_f'] = data.get('pre_f', [])
    return data
