import numpy as np
from utils import *

def shaky_prepend(x, y, H, G, epsilon, sigma, max_updates=100, step_size: float = 1.0):
    Update = True
    f = [G[-1], H[-1]]
    Pg = np.array([np.sum(g(x))/x.shape[0] for g in G])
    noisy_epsilon = float(epsilon + np.random.laplace(0, sigma))

    while Update and len(f)*2 < max_updates:
        Update = False

        # Compute current group losses once
        loss_f_by_g = np.array(get_group_loss(f, x, y, G, step_size))  # shape (len(G),)

        # Compute losses for each h across all g: shape (len(H), len(G))
        losses_h_by_g = np.array([get_group_loss(h, x, y, G, step_size) for h in H])

        # Sample one noise per (h,g)
        mu = np.random.laplace(0, sigma, size=losses_h_by_g.shape)

        # Noisy weighted improvement for each (h,g)
        improvements = (loss_f_by_g[np.newaxis, :] - losses_h_by_g) * Pg[np.newaxis, :] - (noisy_epsilon + mu)

        # Choose a random (h,g) among positive improvements
        positive_pairs = np.argwhere(improvements > 0)
        if positive_pairs.size > 0:
            choice_idx = np.random.randint(positive_pairs.shape[0])
            h_idx, g_idx = positive_pairs[choice_idx]
            print('(', int(g_idx), ',', int(h_idx), ')', end=' ')
            f.append(G[int(g_idx)])
            f.append(H[int(h_idx)])
            Update = True
            noisy_epsilon = float(epsilon + np.random.laplace(0, sigma))

    print(f'shaky prepend with {epsilon}--------------------------------')
    return f

def shaky_prepend_v0(x, y, H, G, epsilon, sigma, max_updates=100, step_size: float = 1.0):
    Update = True
    f = [G[-1], H[-1]]
    Pg = np.array([np.sum(g(x))/x.shape[0] for g in G])
    noisy_epsilon = float(epsilon + np.random.laplace(0, sigma))

    while Update and len(f)*2 < max_updates:
        Update = False

        # Compute current group losses once
        loss_f_by_g = np.array(get_group_loss(f, x, y, G, step_size))  # shape (len(G),)

        # Compute losses for each h across all g: shape (len(H), len(G))
        losses_h_by_g = np.array([get_group_loss(h, x, y, G, step_size) for h in H])

        # Sample one noise per (h,g)
        mu = np.random.laplace(0, sigma, size=losses_h_by_g.shape)

        # Noisy weighted improvement for each (h,g)
        improvements = (loss_f_by_g[np.newaxis, :] - losses_h_by_g) * Pg[np.newaxis, :] - (noisy_epsilon + mu)

        # Choose the best (h,g)
        h_idx, g_idx = np.unravel_index(np.argmax(improvements), improvements.shape)
        max_improvement = improvements[h_idx, g_idx]

        if max_improvement > 0:
            print('(', g_idx, ',', h_idx, ')', end=' ')
            f.append(G[g_idx])
            f.append(H[h_idx])
            Update = True
            noisy_epsilon = float(epsilon + np.random.laplace(0, sigma))

    print(f'shaky prepend with {epsilon}--------------------------------')
    return f