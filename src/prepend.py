import numpy as np
from utils import *

def prepend_constant_epsilon(x, y, H, G, epsilon, max_updates=1000, step_size: float = 1.0):
    min_loss = []  # minimum achievable loss over all experts H for each group g in G
    min_loss_idx = []  # argmin expert index for each group
    for gi in range(len(G)):
        mask = G[gi](x)
        denom = np.sum(mask)
        losses_h = [np.sum((y - h(x))**2 * mask) / denom for h in H]
        best_idx = int(np.argmin(losses_h))
        min_loss_idx.append(best_idx)
        min_loss.append(float(losses_h[best_idx]))
    Update = True
    # print(min_loss)
    f = [G[-1], H[-1]]

    while Update and len(f)*2 < max_updates:
        Update = False
        loss_list = get_group_loss(f, x, y, G, step_size)
        # print(np.array(loss_list) - np.array(min_loss))
        g_index = np.argmax(np.array(loss_list) - np.array(min_loss))
        # print(g_index, loss_list[g_index]- min_loss[g_index], epsilon)
        if loss_list[g_index] > min_loss[g_index] + epsilon:
            f.append(G[g_index])
            f.append(H[min_loss_idx[g_index]])
            Update = True
            print('(', g_index, ',', min_loss_idx[g_index], ')', end=' ')
    print(f'prepend with {epsilon}--------------------------------')
    return f

def prepend_group_epsilon(x, y, H, G, epsilon, max_updates=1000, step_size: float = 1.0):
    Update = True
    f = [G[-1], H[-1]]
    Pg = [np.sum(g(x))/x.shape[0] for g in G]
    min_loss = []  # minimum achievable loss over all experts H for each group g in G
    min_loss_idx = []  # argmin expert index for each group
    for gi in range(len(G)):
        mask = G[gi](x)
        denom = np.sum(mask)
        losses_h = [np.sum((y - h(x))**2 * mask) / denom for h in H]
        best_idx = int(np.argmin(losses_h))
        min_loss_idx.append(best_idx)
        min_loss.append(float(losses_h[best_idx]))

    while Update and len(f)*2 < max_updates:
        Update = False
        loss_list = get_group_loss(f, x, y, G, step_size)
        g_index = np.argmax(Pg*(np.array(loss_list) - np.array(min_loss)))
        if Pg[g_index]*(loss_list[g_index]-min_loss[g_index]) > epsilon:
            f.append(G[g_index])
            f.append(H[min_loss_idx[g_index]])
            Update = True
            print('(', g_index, ',', min_loss_idx[g_index], ')', end=' ')
    print(f'group prepend with {epsilon}--------------------------------')
    return f