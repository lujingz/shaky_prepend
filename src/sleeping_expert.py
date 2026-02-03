import numpy as np

def sleeping_expert(x, y, H, G, factor):
    m = len(H)
    k = len(G)
    n = len(x)

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Evaluate group indicators for all x: shape (k, n)
    G_mask = np.vstack([np.asarray(g(x_arr)).astype(float).reshape(-1) for g in G])

    # Proportion of active points per group and learning rates per group
    Pg = G_mask.sum(axis=1) / n
    eta = np.minimum(1.0, np.sqrt(np.log(m * k) / (n*Pg))) * factor

    # Evaluate predictions of all experts for all x: shape (m, n)
    H_pred = np.vstack([np.asarray(h(x_arr)).reshape(-1) for h in H])

    # Initialize weights
    w = np.full((m, k), 1.0 / (m * k), dtype=float)

    one_minus_exp_neg_eta = 1.0 - np.exp(-eta)  # shape (k,)

    for i in range(n):
        # Active groups for this x_i
        active_g = G_mask[:, i]  # shape (k,)

        # Normalization term p_sum
        factor_g = active_g * one_minus_exp_neg_eta  # shape (k,)
        W_factor = w * factor_g[np.newaxis, :]       # shape (m, k)
        p_sum = W_factor.sum()
        if p_sum <= 0:
            continue

        # Loss per expert at x_i
        r_h = (H_pred[:, i] - y_arr[i]) ** 2  # shape (m,)

        # Expected loss l
        l = (W_factor * r_h[:, np.newaxis]).sum() / p_sum

        # Weight update (vectorized over h and g)
        update = eta[np.newaxis, :] * active_g[np.newaxis, :] * (
            l * np.exp(-eta)[np.newaxis, :] - r_h[:, np.newaxis]
        )
        w *= np.exp(update)

    return eta, w