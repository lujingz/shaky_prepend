import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from src.shaky_prepend import shaky_prepend
from src.prepend import prepend_constant_epsilon, prepend_group_epsilon
from utils import *
from src.sleeping_expert import sleeping_expert

def generate_data_piecewise_and_get_H(n=200, centers=20, intervals=10, noise_level=0):
    # for spatial experiment and fractional shaky prepend experiment
    x = np.linspace(0, 1, n)
    y = np.piecewise(x, [x<0.5, (x >= 0.5) & (x <= 0.75), (x >= 0.75) & (x <= 0.9), (x >= 0.9) & (x <= 1)], [lambda x: 0, lambda x: 0.25, lambda x: 1, lambda x: 0.5])
    y = y + np.random.normal(0, 1, n)*noise_level
    ground_truth = [(lambda x: 0, lambda x: x<0.5), (lambda x: 0.25, lambda x: (x >= 0.5) & (x <= 0.75)), (lambda x: 1, lambda x: (x >= 0.75) & (x <= 0.9)), (lambda x: 0.5, lambda x: (x >= 0.9) & (x <= 1))]
    G_temp = []
    length_interval = np.linspace(0, 1, intervals+1)
    for i in range(centers):
        for j in range(intervals):
            G_temp.append(lambda x, i=i, j=j: (x <= i/centers+length_interval[j]/2) & (x > i/centers-length_interval[j]/2))
    G_temp.append(lambda x: x>=0)
    G = []
    for g in G_temp:
        mask = g(x)
        if np.sum(mask) > 0:
            G.append(g)
    H = []
    for g in G:
        mask = g(x)
        if np.sum(mask) > 0:
            const_value = float(np.mean(y[mask]))
            H.append(lambda x, const_value=const_value: np.full_like(np.array(x), const_value, dtype=float))
    x = np.linspace(0, 1, n)
    y = np.piecewise(x, [x<0.5, (x >= 0.5) & (x <= 0.75), (x >= 0.75) & (x <= 0.9), (x >= 0.9) & (x <= 1)], [lambda x: 0, lambda x: 0.25, lambda x: 1, lambda x: 0.5])
    y = y + np.random.normal(0, 1, n)*noise_level
    return H, G, x, y, ground_truth

def generate_criterion_selection_data_and_get_H(n=2):
    # for criterion selection experiment
    x1 = np.linspace(0,1,10*n)
    y1 = 2.2 + np.random.normal(0, 1, 10*n)*0.3
    x2 = np.random.uniform(1,2,n)
    y2 = 2.2 + np.random.normal(0, 1, n)*0.3
    x3 = np.random.uniform(2,3,n)
    y3 = 5 + np.random.normal(0, 1, n)*0.3
    x4 = np.random.uniform(3,4,3*n)
    y4 = 4 + np.random.normal(0, 1, 3*n)*0.3
    x5 = np.random.uniform(4,5,10*n)
    y5 = 3.5 + np.random.normal(0, 1, 10*n)*0.3 
    x6 = np.random.uniform(5,6,40*n)
    y6 = 4.1 + np.random.normal(0, 1, 40*n)*0.3
    x_full = np.concatenate([x1, x2, x3, x4, x5, x6])
    y_full = np.concatenate([y1, y2, y3, y4, y5, y6])
    m1 = float(np.mean(np.concatenate([y1, y2])))
    h1 = lambda x, m1=m1: np.full_like(np.array(x), m1, dtype=float)
    m2 = float(np.mean(np.concatenate([y2, y3])))
    h2 = lambda x, m2=m2: np.full_like(np.array(x), m2, dtype=float)
    m3 = float(np.mean(np.concatenate([y3, y5])))
    h3 = lambda x, m3=m3: np.full_like(np.array(x), m3, dtype=float)
    m4 = float(np.mean(y_full))
    h4 = lambda x, m4=m4: np.full_like(np.array(x), m4, dtype=float)
    H = [h1, h2, h3, h4]

    # Define groups to align with the trained experts
    g1 = lambda x: np.array(x) <= 2
    g2 = lambda x: (np.array(x) > 1) & (np.array(x) <= 3)
    # Match model3's training support: union of (2,3] and (4,5]
    g3 = lambda x: ((np.array(x) > 2) & (np.array(x) <= 3)) | ((np.array(x) > 4) & (np.array(x) <= 5))
    # Catch-all
    g4 = lambda x: np.array(x) >= 0
    G = [g1, g2, g3, g4]

    # Ground truth piecewise constants
    ground_truth = [
        (lambda x: 2,  lambda x: np.array(x) <= 1),
        (lambda x: 2,  lambda x: (np.array(x) > 1) & (np.array(x) <= 2)),
        (lambda x: 5,  lambda x: (np.array(x) > 2) & (np.array(x) <= 3)),
        (lambda x: 4,  lambda x: (np.array(x) > 3) & (np.array(x) <= 4)),
        (lambda x: 3.5,  lambda x: (np.array(x) > 4) & (np.array(x) <= 5)),
        (lambda x: 4.1,  lambda x: np.array(x) > 5),
    ]

    x1 = np.linspace(0,1,10*n)
    y1 = 2.2 + np.random.normal(0, 1, 10*n)*0.3
    x2 = np.random.uniform(1,2,n)   
    y2 = 2.2 + np.random.normal(0, 1, n)*0.3 # 2
    x3 = np.random.uniform(2,3,n)
    y3 = 5 + np.random.normal(0, 1, n)*0.3
    x4 = np.random.uniform(3,4,3*n)
    y4 = 4 + np.random.normal(0, 1, 3*n)*0.3 # 2
    x5 = np.random.uniform(4,5,10*n)
    y5 = 3.5 + np.random.normal(0, 1, 10*n)*0.3
    x6 = np.random.uniform(5,6,40*n)
    y6 = 4.1 + np.random.normal(0, 1, 40*n)*0.3 
    x_full = np.concatenate([x1, x2, x3, x4, x5, x6])
    y_full = np.concatenate([y1, y2, y3, y4, y5, y6])
    return H, G, x_full, y_full, ground_truth

def generate_data_and_get_H_unbalanced(n=2):
    # for unbalanced group selection experiment
    x1 = np.linspace(0,1,4*n)
    y1 = 0 + np.random.normal(0, 1, 4*n)*0.1
    x2 = np.linspace(1,1.25,n)
    y2 = 0.2 + np.random.normal(0, 1, n)*0.1
    x3 = np.linspace(1.25,1.5,n)
    y3 = 0.2 + np.random.normal(0, 1, n)*0.1
    x4 = np.linspace(1.5,1.75,n)
    y4 = 0.2 + np.random.normal(0, 1, n)*0.1
    x5 = np.linspace(1.75,2,n)
    y5 = 0.2 + np.random.normal(0, 1, n)*0.1
    y_full = np.concatenate([y1, y2, y3, y4, y5])
    
    g1 = lambda x: x<=1
    h1 = lambda x: np.full_like(x, np.mean(y1))
    g2 = lambda x: (x>1) & (x<=1.25)
    h2 = lambda x: np.full_like(x, np.mean(y2))
    g3 = lambda x: (x>1.25) & (x<=1.5)
    h3 = lambda x: np.full_like(x, np.mean(y3))
    g4 = lambda x: (x>1.5) & (x<=1.75)
    h4 = lambda x: np.full_like(x, np.mean(y4))
    g5 = lambda x: (x>1.75) & (x<=2)
    h5 = lambda x: np.full_like(x, np.mean(y5))
    g6 = lambda x: x>1
    h6 = lambda x: np.full_like(x, np.mean(np.concatenate([y2, y3, y4, y5])))
    g7 = lambda x: x>=0
    h7 = lambda x: np.full_like(x, np.mean(np.concatenate([y1, y2, y3, y4, y5])))
    H = [h1, h2, h3, h4, h5, h6, h7]
    G = [g1, g2, g3, g4, g5, g6, g7]
    ground_truth = [(lambda x: 0, lambda x: x<=1), (lambda x: 0.2, lambda x: (x>=1) & (x<=2))]
    x_full = np.concatenate([x1, x2, x3, x4, x5])
    y_new = [0 + np.random.normal(0, 1, 4*n)*0.1, 0.2 + np.random.normal(0, 1, 4*n)*0.1]
    # y_full = np.concatenate(y_new)
    return H, G, x_full, y_full, ground_truth

def tune_hyperparameter(method, x_full, y_full, x_validation, y_validation, H, G, step_size, loss='total_loss', seed_base=None):
    # select the best hyperparameter according to requested loss
    Pg = np.array([np.sum(g(x_full)) / len(x_full) for g in G])
    epsilons = [0.1, 0.01, 0.001, 0.0001]
    # epsilons = [1, 0.1, 0.01]
    # Allow tuning over a list of step sizes
    step_sizes = step_size if isinstance(step_size, list) else [step_size]
    step_size_is_list = isinstance(step_size, list)
    if method == 'prepend':
        best_val = None
        best_epsilon = None
        best_step = None
        for s in step_sizes:
            for epsilon in epsilons:
                _, _, test_loss = run_experiment('prepend', x_full, y_full, x_validation, y_validation, H, G, step_size=s, epsilon=epsilon)
                tl = np.array(test_loss)
                if loss == 'total_loss':
                    metric = tl[-1]
                elif loss == 'average_weighted_loss':
                    metric = float(np.mean(tl * Pg))
                elif loss == 'max_weighted_loss':
                    metric = float(np.max(tl * Pg))
                elif loss == 'worst_loss':
                    metric = float(np.max(tl))
                else:
                    metric = tl[-1]
                if (best_val is None) or (metric < best_val):
                    best_val = metric
                    best_epsilon = epsilon
                    best_step = s
        return (best_epsilon, best_step) if step_size_is_list else best_epsilon
    elif method == 'shaky_prepend':
        best_val = None
        best_epsilon = None
        best_sigma = None
        best_step = None
        for s in step_sizes:
            for epsilon in epsilons:
                sigma = epsilon / 10
                _, _, test_loss = run_experiment('shaky_prepend', x_full, y_full, x_validation, y_validation, H, G, step_size=s, epsilon=epsilon, sigma=sigma)
                tl = np.array(test_loss)
                if loss == 'total_loss':
                    metric = tl[-1]
                elif loss == 'average_weighted_loss':
                    metric = float(np.mean(tl * Pg))
                elif loss == 'max_weighted_loss':
                    metric = float(np.max(tl * Pg))
                elif loss == 'worst_loss':
                    metric = float(np.max(tl))
                else:
                    metric = tl[-1]
                if (best_val is None) or (metric < best_val):
                    best_val = metric
                    best_epsilon = epsilon
                    best_sigma = sigma
                    best_step = s
        return (best_epsilon, best_sigma, best_step) if step_size_is_list else (best_epsilon, best_sigma)
    elif method == 'group_prepend':
        best_val = None
        best_epsilon = None
        best_step = None
        for s in step_sizes:
            for epsilon in epsilons:
                _, _, test_loss = run_experiment('group_prepend', x_full, y_full, x_validation, y_validation, H, G, step_size=s, epsilon=epsilon)
                tl = np.array(test_loss)
                if loss == 'total_loss':
                    metric = tl[-1]
                elif loss == 'average_weighted_loss':
                    metric = float(np.mean(tl * Pg))
                elif loss == 'max_weighted_loss':
                    metric = float(np.max(tl * Pg))
                elif loss == 'worst_loss':
                    metric = float(np.max(tl))
                else:
                    metric = tl[-1]
                if (best_val is None) or (metric < best_val):
                    best_val = metric
                    best_epsilon = epsilon
                    best_step = s
        return (best_epsilon, best_step) if step_size_is_list else best_epsilon
    elif method == 'sleeping_expert':
        # Do not tune step_size for sleeping_expert; use provided value (or first if list)
        factors = [1, 10, 50, 100]
        s = step_size if not isinstance(step_size, list) else step_size[0]
        test_losses = []
        for idx, factor in enumerate(factors):
            # Ensure deterministic evaluation per factor within a run
            seed = None if seed_base is None else (seed_base * 1000 + idx)
            _, _, _, test_loss = run_experiment('sleeping_expert', x_full, y_full, x_validation, y_validation, H, G, step_size=s, seed=seed, factor=factor)
            tl = np.array(test_loss)
            if loss == 'total_loss':
                metric = tl[-1]
            elif loss == 'average_weighted_loss':
                metric = float(np.mean(tl * Pg))
            elif loss == 'max_weighted_loss':
                metric = float(np.max(tl * Pg))
            elif loss == 'worst_loss':
                metric = float(np.max(tl))
            else:
                metric = tl[-1]
            test_losses.append(metric)
        return factors[np.argmin(test_losses)]

def run_experiment(method, x_full, y_full, x_test, y_test, H, G, step_size=1, seed=None, **kwargs):
    if method == 'prepend':
        f = prepend_constant_epsilon(x_full, y_full, H, G, kwargs['epsilon'], step_size=step_size)
        train_loss = get_group_loss(f, x_full, y_full, G, step_size)
        test_loss = get_group_loss(f, x_test, y_test, G, step_size)
    elif method == 'shaky_prepend':
        f = shaky_prepend(x_full, y_full, H, G, kwargs['epsilon'], kwargs['sigma'], step_size=step_size)
        train_loss = get_group_loss(f, x_full, y_full, G, step_size)
        test_loss = get_group_loss(f, x_test, y_test, G, step_size)
    elif method == 'sleeping_expert':
        # Seed before evaluation to strictly control stochastic sampling
        if seed is not None:
            np.random.seed(seed)
        eta, f = sleeping_expert(x_full, y_full, H, G, kwargs['factor'])
        train_loss = get_group_loss_sleeping_expert(f, x_full, y_full, H, G, eta)
        # Use a different deterministic stream for test to decouple from train
        if seed is not None:
            np.random.seed(seed + 1)
        test_loss = get_group_loss_sleeping_expert(f, x_test, y_test, H, G, eta)
        return eta, f, train_loss, test_loss
    elif method == 'group_prepend':
        f = prepend_group_epsilon(x_full, y_full, H, G, kwargs['epsilon'], step_size=step_size)
        train_loss = get_group_loss(f, x_full, y_full, G, step_size)
        test_loss = get_group_loss(f, x_test, y_test, G, step_size)

    return f, train_loss, test_loss

def run_experiment_multiple_times(generate_data_func, num_runs=10, step_size=1, experiment_name="experiment", hp_loss='total_loss', plot=False,**kwargs):
    """Run the experiment multiple times with different random seeds and calculate mean losses."""
    train_losses_prepend = []
    test_losses_prepend = []
    num_updates_prepend = []
    train_losses_prepend_group = []
    test_losses_prepend_group = []
    num_updates_prepend_group = []
    train_losses_shaky = []
    test_losses_shaky = []
    num_updates_shaky = []
    train_losses_best_h = []
    test_losses_best_h = []
    train_losses_sleeping_expert = []
    test_losses_sleeping_expert = []
    num_updates_sleeping_expert = []
    n = sum([i for i in kwargs.values() if isinstance(i, int)])
    for run in range(num_runs):
        # Set different random seed for each run
        np.random.seed(run)
        epsilon = 0.1/(50*n**0.4)
        H, G, x_full, y_full, ground_truth = generate_data_func(**kwargs)
        _, _, x_validation, y_validation, _ = generate_data_func(**kwargs)
        _, _, x_test, y_test, _ = generate_data_func(**kwargs)

        # the best h in H for each g in G
        loss_best_h_train, loss_best_h_test = [], []
        for i in range(len(G)):
            loss_best_h_train.append(np.sum((y_full - H[i](x_full))**2*G[i](x_full))/np.sum(G[i](x_full)))
            loss_best_h_test.append(np.sum((y_test - H[i](x_test))**2*G[i](x_test))/np.sum(G[i](x_test)))
        train_losses_best_h.append(loss_best_h_train)
        test_losses_best_h.append(loss_best_h_test)
        
        # Run prepend algorithm
        ret_prepend = tune_hyperparameter('prepend', x_full, y_full, x_validation, y_validation, H, G, step_size, loss=hp_loss)
        if isinstance(step_size, list):
            epsilon, step_size_prepend = ret_prepend
        else:
            epsilon = ret_prepend
            step_size_prepend = step_size
        f_prepend, train_loss_prepend, test_loss_prepend = run_experiment('prepend', x_full, y_full, x_test, y_test, H, G, step_size=step_size_prepend, epsilon=epsilon)
        train_losses_prepend.append(train_loss_prepend)
        num_updates_prepend.append(len(f_prepend)/2)
        test_losses_prepend.append(test_loss_prepend)
        
        # Run prepend with group epsilon strategy
        ret_group = tune_hyperparameter('group_prepend', x_full, y_full, x_validation, y_validation, H, G, step_size, loss=hp_loss)
        if isinstance(step_size, list):
            epsilon, step_size_group = ret_group
        else:
            epsilon = ret_group
            step_size_group = step_size
        f_group_prepend, train_loss_group_prepend, test_loss_group_prepend = run_experiment('group_prepend', x_full, y_full, x_test, y_test, H, G, step_size=step_size_group, epsilon=epsilon)
        train_losses_prepend_group.append(train_loss_group_prepend)
        num_updates_prepend_group.append(len(f_group_prepend)/2)
        test_losses_prepend_group.append(test_loss_group_prepend)

        # Run shaky prepend algorithm
        ret_shaky = tune_hyperparameter('shaky_prepend', x_full, y_full, x_validation, y_validation, H, G, step_size, loss=hp_loss)
        if isinstance(step_size, list):
            epsilon, sigma, step_size_shaky = ret_shaky
        else:
            epsilon, sigma = ret_shaky
            step_size_shaky = step_size
        f_shaky, train_loss_shaky, test_loss_shaky = run_experiment('shaky_prepend', x_full, y_full, x_test, y_test, H, G, step_size=step_size_shaky, epsilon=epsilon, sigma=sigma)
        train_losses_shaky.append(train_loss_shaky)
        num_updates_shaky.append(len(f_shaky)/2)
        test_losses_shaky.append(test_loss_shaky)

        # Run sleeping expert algorithm (no step_size tuning)
        factor = tune_hyperparameter('sleeping_expert', x_full, y_full, x_validation, y_validation, H, G, step_size, loss=hp_loss, seed_base=run)
        step_size_sleep = step_size if not isinstance(step_size, list) else step_size[0]
        eta, f_sleeping_expert, train_loss_sleeping_expert, test_loss_sleeping_expert = run_experiment('sleeping_expert', x_full, y_full, x_test, y_test, H, G, step_size=step_size_sleep, seed=run, factor=factor)
        train_losses_sleeping_expert.append(train_loss_sleeping_expert)
        num_updates_sleeping_expert.append(len(f_sleeping_expert)/2)
        test_losses_sleeping_expert.append(test_loss_sleeping_expert)

        # Plot fitting and data points for the first run
        if run == 0 and plot:
            plot_fitting_and_data(x_full, y_full, f_prepend, f_group_prepend, f_shaky, f_sleeping_expert, H, G, eta, experiment_name, ground_truth, step_size=step_size_prepend)
            # Additional figure: group prepend vs ground truth with data points
            plot_group_prepend_and_ground_truth_and_data(x_full, y_full, f_group_prepend, G, experiment_name + "_group_prepend_only", ground_truth, step_size=step_size_group)
            plot_group_prepend_and_ground_truth_and_data(x_full, y_full, f_prepend, G, experiment_name + "_prepend_only", ground_truth, step_size=step_size_prepend, fit_label='Prepend Fitting')
            plot_group_prepend_and_ground_truth_and_data(x_full, y_full, f_shaky, G, experiment_name + "_shaky_prepend_only", ground_truth, step_size=step_size_shaky, fit_label='Shaky Prepend Fitting')
            plot_sleeping_expert_and_ground_truth_and_data(x_full, y_full, f_sleeping_expert, H, G, eta, experiment_name + "_sleeping_expert_only", ground_truth)
            
    # Calculate mean losses across all runs
    mean_test_loss_prepend = np.mean(test_losses_prepend, axis=0)
    mean_num_updates_prepend = np.mean(num_updates_prepend, axis=0)
    mean_test_loss_prepend_group = np.mean(test_losses_prepend_group, axis=0)
    mean_num_updates_prepend_group = np.mean(num_updates_prepend_group, axis=0)
    mean_test_loss_shaky = np.mean(test_losses_shaky, axis=0)
    mean_num_updates_shaky = np.mean(num_updates_shaky, axis=0)
    mean_num_updates_sleeping_expert = np.mean(num_updates_sleeping_expert, axis=0)
    mean_test_loss_sleeping_expert = np.mean(test_losses_sleeping_expert, axis=0)

    # the worst, the average across group, total loss, the loss of the smallest group
    Pg = [np.sum(g(x_full))/x_full.shape[0] for g in G]
    print("Number of updates for methods: prepend, prepend_group, shaky_prepend, sleeping_expert")
    print(mean_num_updates_prepend, mean_num_updates_prepend_group, mean_num_updates_shaky, mean_num_updates_sleeping_expert)
    per_run_test_losses = {
        'prepend': np.array(test_losses_prepend),
        'prepend_group': np.array(test_losses_prepend_group),
        'shaky_prepend': np.array(test_losses_shaky),
        'sleeping_expert': np.array(test_losses_sleeping_expert),
    }
    per_run_updates = {
        'prepend': np.array(num_updates_prepend),
        'prepend_group': np.array(num_updates_prepend_group),
        'shaky_prepend': np.array(num_updates_shaky),
        'sleeping_expert': np.array(num_updates_sleeping_expert),
    }
    return {
        'per_run_test_losses': per_run_test_losses,
        'per_run_updates': per_run_updates,
        'experiment_name': experiment_name
    }
    
def main():
    #----------------------criterion selection experiments----------------------
    # big sample size
    result_total = run_experiment_multiple_times(generate_criterion_selection_data_and_get_H, 20, 1, "criteria_selection_total_loss", n=400)
    result_worst = run_experiment_multiple_times(generate_criterion_selection_data_and_get_H, 20, 1, "criteria_selection_worst_loss", hp_loss='worst_loss', n=400)
    errorplot(result_total, result_worst, output_name='errorplot_criterion_total_loss_400', loss_type='total_loss', label1_suffix='total', label2_suffix='worst', sleeping_expert=True)
    errorplot(result_total, result_worst, output_name='errorplot_criterion_worst_loss_400', loss_type='worst_loss', label1_suffix='total', label2_suffix='worst', sleeping_expert=True)
    # small sample size
    result_total = run_experiment_multiple_times(generate_criterion_selection_data_and_get_H, 20, 1, "criteria_selection_total_loss", n=4)
    result_worst = run_experiment_multiple_times(generate_criterion_selection_data_and_get_H, 20, 1, "criteria_selection_worst_loss", hp_loss='worst_loss', n=4)
    errorplot(result_total, result_worst, output_name='errorplot_criterion_total_loss_4', loss_type='total_loss', label1_suffix='total', label2_suffix='worst', sleeping_expert=True)
    errorplot(result_total, result_worst, output_name='errorplot_criterion_worst_loss_4', loss_type='worst_loss', label1_suffix='total', label2_suffix='worst', sleeping_expert=True)

    #----------------------unbalanced group selection experiments----------------------
    result_total = run_experiment_multiple_times(generate_data_and_get_H_unbalanced, 20, 1, "unbalanced", n=15)
    errorplot_single(result_total, output_name='errorplot_unbalanced_experiment')
    
    #----------------------fractional shaky prepend experiments----------------------
    result1 = run_experiment_multiple_times(generate_data_piecewise_and_get_H, 20, 1, "piecewise_experiment_noise", centers=20, intervals=20)
    result2 = run_experiment_multiple_times(generate_data_piecewise_and_get_H, 20, [1,0.5], "piecewise_experiment_noise_1_0.5", centers=20, intervals=20)
    errorplot(result1, result2, output_name='errorplot_piecewise_experiment', loss_type='total_loss', label1_suffix='original', label2_suffix='fractional')
    errorplot(result1, result2, output_name='errorplot_piecewise_experiment_worst_loss', loss_type='worst_loss', label1_suffix='original', label2_suffix='fractional')

    #----------------------spatial adaptivity experiments----------------------
    result = run_experiment_multiple_times(generate_data_piecewise_and_get_H, 20, 1, "spatial_adaptivity_noisy", centers=20, intervals=20, plot=True, noise_level=0.1)
    result = run_experiment_multiple_times(generate_data_piecewise_and_get_H, 20, 1, "spatial_adaptivity", centers=20, intervals=20, plot=True, noise_level=0)

if __name__ == "__main__":
    main()