from __future__ import annotations
import csv
import json
import time
from functools import partial
from pathlib import Path
import numpy as np
from mechanics import PROBLEM_SPECS, load_model, levels_to_q, levels_to_free_u, batch_objective_log, make_train_validation_solver, make_train_validation_sweep

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0.0:
        return float('nan')
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)

def train_with_jax_solver(dataset_x, dataset_t_std, free_dim, num_bd, lambda_sr, learning_rate, num_epoch):
    import jax
    import jax.numpy as jnp
    import optax
    x = jnp.asarray(dataset_x, dtype=jnp.float32)
    t = jnp.asarray(dataset_t_std, dtype=jnp.float32)
    channels = free_dim * num_bd
    rank = 4
    key1, key2 = jax.random.split(jax.random.PRNGKey(173))
    del key1
    glorot = jax.nn.initializers.glorot_normal()
    params = {'V': glorot(key2, (channels, rank), jnp.float32), 'b': jnp.zeros((channels,), dtype=jnp.float32), 'a': jnp.zeros((1,), dtype=jnp.float32)}
    optimizer = optax.amsgrad(learning_rate)
    opt_state = optimizer.init(params)

    def predict(params_, x_):
        w = params_['V'] @ params_['V'].T
        return params_['a'] + jnp.sum(params_['b'].reshape(1, channels) * x_, axis=1) + jnp.sum(x_.reshape(x_.shape[0], 1, channels) * x_.reshape(x_.shape[0], channels, 1) * w.reshape(1, channels, channels), axis=(1, 2))

    def loss(params_):
        y = predict(params_, x)
        l1 = jnp.mean((t - y) ** 2)
        v = params_['V'].reshape(free_dim, num_bd, rank)
        b = params_['b'].reshape(free_dim, num_bd)
        l2 = jnp.sum((v[:, 1:, :] - v[:, :-1, :]) ** 2) + jnp.sum((b[:, 1:] - b[:, :-1]) ** 2)
        return l1 + lambda_sr * l2
    value_and_grad = jax.value_and_grad(loss)

    def train_step(params_, opt_state_):
        loss_value, grads = value_and_grad(params_)
        updates, opt_state_next = optimizer.update(grads, opt_state_)
        params_next = optax.apply_updates(params_, updates)
        return (params_next, opt_state_next, loss_value)

    def scan_step(carry, _):
        params_, opt_state_ = carry
        params_next, opt_state_next, loss_value = train_step(params_, opt_state_)
        return ((params_next, opt_state_next), loss_value)

    @partial(jax.jit, static_argnames=('steps',))
    def train_steps(params_, opt_state_, steps: int):
        (params_next, opt_state_next), losses = jax.lax.scan(scan_step, (params_, opt_state_), xs=None, length=steps)
        return (params_next, opt_state_next, losses[-1])
    log_every = max(1, num_epoch // 200)
    history = []
    epoch = 0
    while epoch < num_epoch:
        steps = min(log_every, num_epoch - epoch)
        params, opt_state, loss_value = train_steps(params, opt_state, steps)
        epoch += steps
        history.append((epoch, float(loss_value)))
    v = np.asarray(params['V'])
    b = np.asarray(params['b'])
    a = np.asarray(params['a'])
    qij = v @ v.T
    for i in range(channels):
        qij[i, i] += b[i]
    pred_train_std = np.asarray(predict(params, x))
    return (qij, a, pred_train_std, history)

def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def evaluate_qij(q_flat: np.ndarray, qij: np.ndarray, bias: float) -> np.ndarray:
    return np.einsum('bi,ij,bj->b', q_flat, qij, q_flat, optimize=True) + bias

def train_with_jax_sweep(dataset_x, dataset_t_std, free_dim, num_bd, lambda_sr, learning_rate, num_epoch):
    import jax
    import jax.numpy as jnp
    import optax
    x = jnp.asarray(dataset_x, dtype=jnp.float32)
    t = jnp.asarray(dataset_t_std, dtype=jnp.float32)
    channels = free_dim * num_bd
    rank = 4
    key1, key2 = jax.random.split(jax.random.PRNGKey(173))
    del key1
    glorot = jax.nn.initializers.glorot_normal()
    params = {'V': glorot(key2, (channels, rank), jnp.float32), 'b': jnp.zeros((channels,), dtype=jnp.float32), 'a': jnp.zeros((1,), dtype=jnp.float32)}
    optimizer = optax.amsgrad(learning_rate)
    opt_state = optimizer.init(params)

    def predict(params_, x_):
        w = params_['V'] @ params_['V'].T
        return params_['a'] + jnp.sum(params_['b'].reshape(1, channels) * x_, axis=1) + jnp.sum(x_.reshape(x_.shape[0], 1, channels) * x_.reshape(x_.shape[0], channels, 1) * w.reshape(1, channels, channels), axis=(1, 2))

    def loss(params_):
        y = predict(params_, x)
        l1 = jnp.mean((t - y) ** 2)
        v = params_['V'].reshape(free_dim, num_bd, rank)
        b = params_['b'].reshape(free_dim, num_bd)
        l2 = jnp.sum((v[:, 1:, :] - v[:, :-1, :]) ** 2) + jnp.sum((b[:, 1:] - b[:, :-1]) ** 2)
        return l1 + lambda_sr * l2
    value_and_grad = jax.value_and_grad(loss)

    @jax.jit
    def train_step(params_, opt_state_):
        loss_value, grads = value_and_grad(params_)
        updates, opt_state_next = optimizer.update(grads, opt_state_)
        params_next = optax.apply_updates(params_, updates)
        return (params_next, opt_state_next, loss_value)
    log_every = max(1, num_epoch // 200)
    history = []
    for epoch in range(num_epoch):
        params, opt_state, loss_value = train_step(params, opt_state)
        if epoch == 0 or (epoch + 1) % log_every == 0 or epoch + 1 == num_epoch:
            history.append((epoch + 1, float(loss_value)))
    v = np.asarray(params['V'])
    b = np.asarray(params['b'])
    a = np.asarray(params['a'])
    qij = v @ v.T
    for i in range(channels):
        qij[i, i] += b[i]
    pred_train_std = np.asarray(predict(params, x))
    return (qij, a, pred_train_std, history)

def train(problem, outdir, lambda_sr, train_seed=None, epochs=None, family='solver', validation_seed=456, validation_size=None) -> Path:
    make_train_validation = make_train_validation_solver if family == 'solver' else make_train_validation_sweep
    train_with_jax = train_with_jax_solver if family == 'solver' else train_with_jax_sweep
    spec = PROBLEM_SPECS[problem]
    model = load_model(problem)
    free_dim = int(np.sum(model['BC'] == 0))
    epochs = epochs if epochs is not None else spec.default_epochs
    if train_seed is None:
        train_seed = 321 if problem in {'problem_1', 'problem_3'} else 123
    if validation_size is None:
        validation_size = min(3600, spec.num_bd ** free_dim - spec.num_dataset)
    train_levels, validation_levels = make_train_validation(free_dim=free_dim, num_bd=spec.num_bd, num_train=spec.num_dataset, num_validation=validation_size, train_seed=train_seed, validation_seed=validation_seed)
    train_q = levels_to_q(train_levels, spec.num_bd).reshape(spec.num_dataset, free_dim * spec.num_bd)
    train_u = levels_to_free_u(train_levels, spec.num_bd, spec.ran, spec.free_shift)
    train_y = batch_objective_log(model, train_u)
    target_mean = float(np.mean(train_y))
    target_sd = float(np.std(train_y))
    train_y_std = (train_y - target_mean) / target_sd
    start = time.perf_counter()
    qij, bias, pred_train_std, history = train_with_jax(train_q, train_y_std, free_dim=free_dim, num_bd=spec.num_bd, lambda_sr=lambda_sr, learning_rate=spec.learning_rate, num_epoch=epochs)
    elapsed = time.perf_counter() - start
    validation_q = levels_to_q(validation_levels, spec.num_bd).reshape(len(validation_levels), free_dim * spec.num_bd)
    validation_u = levels_to_free_u(validation_levels, spec.num_bd, spec.ran, spec.free_shift)
    validation_y = batch_objective_log(model, validation_u)
    pred_train = pred_train_std * target_sd + target_mean
    pred_validation = evaluate_qij(validation_q, qij, float(bias[0])) * target_sd + target_mean
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f'{spec.input_name}_Qijbias_fsrfm_std_{lambda_sr}_{spec.num_dataset}_{spec.learning_rate}_{epochs}'
    npz_path = outdir / f'{stem}.npz'
    np.savez(npz_path, Qij=qij, bias=bias, target_mean=target_mean, target_sd=target_sd, train_levels=train_levels, validation_levels=validation_levels, train_y=train_y, validation_y=validation_y, pred_train=pred_train, pred_validation=pred_validation)
    metrics = {'problem': problem, 'lambda_sr': lambda_sr, 'epochs': epochs, 'learning_rate': spec.learning_rate, 'num_train': spec.num_dataset, 'num_validation': len(validation_levels), 'train_seed': train_seed, 'validation_seed': validation_seed, 'train_r2': r2_score(train_y, pred_train), 'validation_r2': r2_score(validation_y, pred_validation), 'target_mean': target_mean, 'target_sd': target_sd, 'elapsed_seconds': elapsed, 'npz': str(npz_path)}
    (outdir / f'{stem}.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    save_csv(outdir / f'{stem}_loss_history.csv', [{'epoch': e, 'loss': v} for e, v in history], ['epoch', 'loss'])
    save_csv(outdir / f'{stem}_metrics.csv', [metrics], list(metrics.keys()))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return npz_path

def all_nontraining_validation(train_levels: np.ndarray, num_bd: int) -> np.ndarray:
    train_set = {tuple(row) for row in np.asarray(train_levels, dtype=int)}
    rows = [[level] for level in range(num_bd) if (level,) not in train_set]
    return np.asarray(rows, dtype=int)

def refresh_problem1_row(row: dict[str, str], model: dict) -> dict[str, str]:
    spec = PROBLEM_SPECS['problem_1']
    npz_path = Path(row['npz'])
    data = np.load(npz_path)
    train_levels = np.asarray(data['train_levels'], dtype=int)
    validation_levels = all_nontraining_validation(train_levels, spec.num_bd)
    validation_q = levels_to_q(validation_levels, spec.num_bd).reshape(len(validation_levels), spec.num_bd)
    validation_u = levels_to_free_u(validation_levels, spec.num_bd, spec.ran, spec.free_shift)
    validation_y = batch_objective_log(model, validation_u)
    pred_validation = evaluate_qij(validation_q, data['Qij'], float(data['bias'][0]))
    pred_validation = pred_validation * float(data['target_sd']) + float(data['target_mean'])
    out = dict(row)
    out['num_validation'] = str(len(validation_levels))
    out['validation_r2'] = repr(r2_score(validation_y, pred_validation))
    return out

