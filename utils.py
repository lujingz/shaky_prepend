import numpy as np
from typing import List
import matplotlib.pyplot as plt

def get_f(f,step_size):
    if isinstance(f, List):
        temp_f = lambda x: f[1](x)
        for i in range(len(f)//2):
            prev = temp_f
            temp_f = lambda x, i=i, prev=prev: (
                (f[2*i](x) * f[2*i+1](x) * step_size) + (1 - f[2*i](x)*step_size) * prev(x)
            )
    else:
        temp_f = f
    return temp_f

def get_loss(f, x, y, g, step_size=1):
    if np.sum(g(x)) == 0:
        return 0
    temp_f = get_f(f,step_size)
    loss = np.sum((y - temp_f(x))**2*g(x))/np.sum(g(x))
    return loss

def get_group_loss(f, x, y, G, step_size=1):
    loss_list = []
    for i in range(len(G)):
        loss_list.append(get_loss(f, x, y, G[i], step_size))
    return loss_list

def get_group_loss_sleeping_expert(w, x, y, H, G, eta):
    y_pred = get_sleeping_expert_predictions(w, x, H, G, eta)
    loss_list = []
    for g in G:
        loss_list.append(np.sum((y - y_pred)**2*g(x))/np.sum(g(x)))
    return loss_list

def get_sleeping_expert_predictions(w, x, H, G, eta):
    # Shapes
    m = len(H)
    k = len(G)
    x_arr = np.asarray(x)

    # Evaluate group indicators for all x: shape (k, n)
    G_mask = np.vstack([np.asarray(g(x_arr)).astype(float).reshape(-1) for g in G])
    one_minus_exp_neg_eta = 1.0 - np.exp(-np.asarray(eta).reshape(-1))  # (k,)

    # Evaluate predictions of all experts for all x: shape (m, n)
    H_pred = np.vstack([np.asarray(h(x_arr)).reshape(-1) for h in H])

    # Compute unnormalized probabilities per expert and x: P = w @ (G_mask * (1-exp(-eta)))
    factor = G_mask * one_minus_exp_neg_eta[:, np.newaxis]  # (k, n)
    P = w @ factor  # (m, n)

    # Normalize across experts per x; fallback to uniform if column sums are zero
    p_sum = P.sum(axis=0)  # (n,)
    P_norm = np.zeros_like(P)
    good = p_sum > 0
    P_norm[:, good] = P[:, good] / p_sum[good]
    if np.any(~good):
        P_norm[:, ~good] = 1.0 / m

    # Vectorized sampling of expert indices per x
    cum_prob = np.cumsum(P_norm, axis=0)  # (m, n)
    # Guard against numeric issues
    cum_prob[-1, :] = 1.0
    u = np.random.rand(x_arr.shape[0])[np.newaxis, :]  # (1, n)
    chosen_idx = (cum_prob >= u).argmax(axis=0)  # (n,)

    # Gather predictions
    y_pred = H_pred[chosen_idx, np.arange(x_arr.shape[0])]
    return y_pred

def doppler_function():
    n_samples = 500
    x = np.linspace(0, 1, n_samples)
    y = np.sqrt(x*(1-x))*np.sin(2.1*np.pi/(x+0.05)) + np.random.normal(0, 1, n_samples) * 0.1
    plt.scatter(x, y)
    plt.show()

def errorplot(results_v1, results_v1_tune, output_name=None, loss_type='total_loss', label1_suffix='fixed', label2_suffix='tune', sleeping_expert=False):
    """
    Variant of ci_visualization: plot mean ± standard error (SE) of losses for each method,
    comparing two result sets (e.g., fixed vs tuned step size).
    If sleeping_expert=True, include the sleeping_expert method as well.
    """
    import os
    os.makedirs('results', exist_ok=True)
    methods = ['prepend', 'prepend_group', 'shaky_prepend']
    if sleeping_expert:
        methods.append('sleeping_expert')
    means_v1, se_v1 = [], []
    means_v2, se_v2 = [], []
    labels = methods
    def mean_and_se(values):
        values = np.asarray(values)
        n = values.shape[0]
        mean_val = float(np.mean(values))
        if n > 1:
            std_val = float(np.std(values, ddof=1))
            se_val = std_val / np.sqrt(n)
        else:
            se_val = 0.0
        return mean_val, se_val
    for m in methods:
        arr_v1 = np.asarray(results_v1['per_run_test_losses'][m])
        arr_v2 = np.asarray(results_v1_tune['per_run_test_losses'][m])
        if loss_type == 'worst_loss':
            vals_v1 = arr_v1.max(axis=1)
            vals_v2 = arr_v2.max(axis=1)
            ylabel = 'Worst loss'
        else:
            vals_v1 = arr_v1[:, -1]
            vals_v2 = arr_v2[:, -1]
            ylabel = 'Total loss'
        m1, s1 = mean_and_se(vals_v1)
        m2, s2 = mean_and_se(vals_v2)
        means_v1.append(m1)
        se_v1.append(s1)
        means_v2.append(m2)
        se_v2.append(s2)
    x = np.arange(len(methods))
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    offset = 0.15
    ax.errorbar(x - offset, means_v1, yerr=se_v1, fmt='o', capsize=5, label=label1_suffix)
    ax.errorbar(x + offset, means_v2, yerr=se_v2, fmt='o', capsize=5, label=label2_suffix)
    ax.set_ylabel(ylabel,fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, fontsize=17)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=15)
    out = output_name or f'{results_v1["experiment_name"]}_vs_{results_v1_tune["experiment_name"]}_SE'
    plt.savefig(f'results/{out}.png', dpi=300, bbox_inches='tight')
    plt.close()

def errorplot_single(results, output_name=None):
    """
    Create a single figure with two panels showing mean ± SE:
    - Left: Total loss (last group's loss) across runs for each method
    - Right: Worst loss (max across groups) across runs for each method
    """
    import os
    os.makedirs('results', exist_ok=True)
    methods = ['prepend', 'prepend_group', 'shaky_prepend', 'sleeping_expert']
    labels = methods
    totals_means, totals_se = [], []
    worst_means, worst_se = [], []
    def mean_and_se(values):
        values = np.asarray(values)
        n = values.shape[0]
        mean_val = float(np.mean(values))
        if n > 1:
            std_val = float(np.std(values, ddof=1))
            se_val = std_val / np.sqrt(n)
        else:
            se_val = 0.0
        return mean_val, se_val
    for m in methods:
        arr = np.asarray(results['per_run_test_losses'][m])
        total_vals = arr[:, -1]
        worst_vals = arr.max(axis=1)
        m_total, se_total = mean_and_se(total_vals)
        m_worst, se_worst = mean_and_se(worst_vals)
        totals_means.append(m_total)
        totals_se.append(se_total)
        worst_means.append(m_worst)
        worst_se.append(se_worst)
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    axes[0].errorbar(x, totals_means, yerr=totals_se, fmt='o', capsize=5)
    axes[0].set_ylabel('Total loss', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, fontsize=17)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].grid(True, axis='y', alpha=0.3)
    axes[1].errorbar(x, worst_means, yerr=worst_se, fmt='o', capsize=5)
    axes[1].set_ylabel('Worst loss', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, fontsize=17)
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].grid(True, axis='y', alpha=0.3)
    out = output_name or f'{results["experiment_name"]}_SE'
    plt.savefig(f'results/{out}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_fitting_and_data(x_full, y_full, f_prepend, f_group_prepend, f_shaky, w_sleeping_expert, H, G, eta, experiment_name, ground_truth=None, step_size=1):
    """Plot the original data points and the fitted curves for prepend, group prepend, shaky prepend, and sleeping expert."""
    plt.figure(figsize=(12, 7))  # Increased figure size
    # plt.scatter(x_full, y_full, color='blue', alpha=0.25, s=28, label='Original Data', zorder=1)  # Slightly lighter and behind

    # Generate points for smooth curves
    x_plot = np.linspace(min(x_full), max(x_full), 200)
    
    # Plot predictions for prepend if available
    if f_prepend is not None:
        y_prepend = get_f(f_prepend, step_size)(x_plot)
        plt.scatter(x_plot, y_prepend, label='Prepend Fitting', alpha=0.6, s=12, c='red', marker='o', zorder=2)
    
    # Plot predictions for group prepend if available
    if f_group_prepend is not None:
        y_group_prepend = get_f(f_group_prepend, step_size)(x_plot)
        plt.scatter(x_plot, y_group_prepend, label='Group Prepend Fitting', alpha=0.6, s=20, c='magenta', marker='x', zorder=2)
    
    # Plot predictions for shaky prepend if available
    if f_shaky is not None:
        y_shaky = get_f(f_shaky, step_size)(x_plot)
        plt.scatter(x_plot, y_shaky, label='Shaky Prepend Fitting', alpha=0.6, s=12, c='green', marker='^', zorder=2)
    
    # Plot predictions for sleeping expert if available
    if w_sleeping_expert is not None:
        y_sleeping_expert = get_sleeping_expert_predictions(w_sleeping_expert, x_plot, H, G, eta)
        plt.scatter(x_plot, y_sleeping_expert, label='Sleeping Expert Fitting', alpha=0.5, s=10, c='blue', marker='.', zorder=2)
    
    if ground_truth:
        y_ground_truth = np.zeros_like(x_plot)
        for f, g in ground_truth:
            mask = g(x_plot)
            y_ground_truth[mask] = f(x_plot[mask])
        plt.scatter(x_plot, y_ground_truth, label='Ground Truth', alpha=0.5, s=10, c='black', zorder=3)
    
    n = x_full.shape[0]
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    # plt.title(f'Data and Fitted Curves - {experiment_name} - n={n}', fontsize=14)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/{experiment_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_group_prepend_and_ground_truth_and_data(x_full, y_full, f_group_prepend, G, experiment_name, ground_truth=None, step_size=1, fit_label='Group Prepend Fitting'):
    """Plot a prepend-style fitted curve (prepend/group/shaky), the ground truth piecewise function, and the original data points."""
    plt.figure(figsize=(12, 7))
    x_plot = np.linspace(min(x_full), max(x_full), 200)
    # scatter original data
    plt.scatter(x_full, y_full, color='blue', alpha=0.25, s=28, label='Original Data', zorder=1)
    if f_group_prepend is not None:
        y_group_prepend = get_f(f_group_prepend, step_size)(x_plot)
        plt.scatter(x_plot, y_group_prepend, label=fit_label, alpha=0.7, s=20, c='magenta', marker='x', zorder=2)
    if ground_truth:
        y_ground_truth = np.zeros_like(x_plot, dtype=float)
        for f, g in ground_truth:
            mask = g(x_plot)
            y_ground_truth[mask] = f(x_plot[mask])
        plt.scatter(x_plot, y_ground_truth, label='Ground Truth', alpha=0.7, s=12, c='black', marker='.', zorder=3)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/{experiment_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_sleeping_expert_and_ground_truth_and_data(x_full, y_full, w_sleeping_expert, H, G, eta, experiment_name, ground_truth=None):
    """Plot the sleeping expert fitted curve, the ground truth piecewise function, and the original data points."""
    plt.figure(figsize=(12, 7))
    x_plot = np.linspace(min(x_full), max(x_full), 200)
    # scatter original data
    plt.scatter(x_full, y_full, color='blue', alpha=0.25, s=28, label='Original Data', zorder=1)
    if w_sleeping_expert is not None:
        y_sleep = get_sleeping_expert_predictions(w_sleeping_expert, x_plot, H, G, eta)
        plt.scatter(x_plot, y_sleep, label='Sleeping Expert Fitting', alpha=0.7, s=12, c='blue', marker='.', zorder=2)
    if ground_truth:
        y_ground_truth = np.zeros_like(x_plot, dtype=float)
        for f, g in ground_truth:
            mask = g(x_plot)
            y_ground_truth[mask] = f(x_plot[mask])
        plt.scatter(x_plot, y_ground_truth, label='Ground Truth', alpha=0.7, s=12, c='black', marker='.', zorder=3)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/{experiment_name}.png', dpi=300, bbox_inches='tight')
    plt.close()