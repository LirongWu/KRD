import copy
import torch
import numpy as np
# from pathlib import Path

from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training for teacher GNNs
def train(model, g, feats, labels, criterion, optimizer, idx):

    model.train()

    logits = model(g, feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx], labels[idx])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# Testing for teacher GNNs
def evaluate(model, g, feats):

    model.eval()

    with torch.no_grad():
        logits = model(g, feats)
        out = logits.log_softmax(dim=1)

    return logits, out


# Training for student MLPs
def train_mini_batch(model, edge_idx, feats, labels, out_t_all, criterion_l, criterion_t, optimizer, idx, param):

    model.train()

    logits = model(None, feats)
    out = logits.log_softmax(dim=1)
    loss_l = criterion_l(out[idx], labels[idx])
    loss_t = criterion_t((logits/param['tau']).log_softmax(dim=1)[edge_idx[0]], (out_t_all/param['tau']).log_softmax(dim=1)[edge_idx[1]])
    loss = loss_l * param['lamb'] + loss_t * (1 - param['lamb'])

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    return loss_l.item() * param['lamb'], loss_t.item() * (1-param['lamb'])


# Testing for student MLPs
def evaluate_mini_batch(model, feats):

    model.eval()

    with torch.no_grad():
        logits = model(None, feats)
        out = logits.log_softmax(dim=1)

    return logits, out


def train_teacher(param, model, g, feats, labels, indices, criterion, evaluator, optimizer):

    if param['exp_setting'] == 'tran':
        idx_train, idx_val, idx_test = indices
    else:
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
        obs_feats = feats[idx_obs]
        obs_labels = labels[idx_obs]
        obs_g = g.subgraph(idx_obs).to(device)

    g = g.to(device)

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0

    for epoch in range(1, param["max_epoch"] + 1):
        if param['exp_setting'] == 'tran':
            train_loss = train(model, g, feats, labels, criterion, optimizer, idx_train)
            _, out = evaluate(model, g, feats)
            train_acc = evaluator(out[idx_train], labels[idx_train])
            val_acc = evaluator(out[idx_val], labels[idx_val])
            test_acc = evaluator(out[idx_test], labels[idx_test])
        else:
            train_loss = train(model, obs_g, obs_feats, obs_labels, criterion, optimizer, obs_idx_train)
            _, obs_out = evaluate(model, obs_g, obs_feats)
            train_acc = evaluator(obs_out[obs_idx_train], obs_labels[obs_idx_train])
            val_acc = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
            _, out = evaluate(model, g, feats)
            test_acc = evaluator(out[idx_test_ind], labels[idx_test_ind])

        if epoch % 1 == 0:
            print("\033[0;30;46m [{}] CLA: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f}\033[0m".format(
                                        epoch, train_loss, train_acc, val_acc, test_acc, val_best, test_val, test_best))

        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            test_val = test_acc
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1
            
        if es == 50:
            print("Early stopping!")
            break


    model.load_state_dict(state)
    model.eval()
    if param['exp_setting'] == 'tran':
        out, _ = evaluate(model, g, feats)
    else:
        obs_out, _ = evaluate(model, obs_g, obs_feats)
        out, _ = evaluate(model, g, feats)
        out[idx_obs] = obs_out

    return out, test_acc, test_val, test_best, state


def train_student(param, model, g, feats, labels, out_t_all, indices, criterion_l, criterion_t, evaluator, optimizer, KD_model):

    if param['exp_setting'] == 'tran':
        idx_train, idx_val, idx_test = indices
        num_node = feats.shape[0]
        edge_idx_list = extract_indices(g)
    else:
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
        obs_feats = feats[idx_obs]
        obs_labels = labels[idx_obs]
        obs_out_t = out_t_all[idx_obs]
        num_node = obs_feats.shape[0]
        edge_idx_list = extract_indices(g.subgraph(idx_obs))

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0

    for epoch in range(1, param["max_epoch"] + 1):

        if epoch == 1:
            edge_idx = edge_idx_list[0]   

        elif (epoch >= 50 and epoch % 10 == 0 and param['dataset'] == 'ogbn-arxiv') or (epoch >= 50 and param['dataset'] != 'ogbn-arxiv' and param['teacher'] != 'GCN') or (epoch > 50 and param['dataset'] != 'ogbn-arxiv' and param['teacher'] == 'GCN'):
            if param['exp_setting'] == 'tran':
                KD_model.updata(out_t_all.detach().cpu().numpy(), logits_s.detach().cpu().numpy())
            else:
                KD_model.updata(obs_out_t.detach().cpu().numpy(), logits_s.detach().cpu().numpy())
            KD_prob = KD_model.predict_prob()
            sampling_mask = torch.bernoulli(torch.tensor(KD_prob[edge_idx_list[1][1]])).bool() 
            edge_idx = torch.masked_select(edge_idx_list[1], sampling_mask).view(2, -1).detach().cpu().numpy().swapaxes(1, 0)

            edge_idx = edge_idx.tolist()
            for i in range(num_node):
                edge_idx.append([i, i])
            edge_idx = np.array(edge_idx).swapaxes(1, 0)            
            

        if param['exp_setting'] == 'tran':
            loss_l, loss_t = train_mini_batch(model, edge_idx, feats, labels, out_t_all, criterion_l, criterion_t, optimizer, idx_train, param)
            logits_s, out = evaluate_mini_batch(model, feats)
            train_acc = evaluator(out[idx_train], labels[idx_train])
            val_acc = evaluator(out[idx_val], labels[idx_val])
            test_acc = evaluator(out[idx_test], labels[idx_test])

        else:
            loss_l, loss_t = train_mini_batch(model, edge_idx, obs_feats, obs_labels, obs_out_t, criterion_l, criterion_t, optimizer, obs_idx_train, param)
            logits_s, obs_out = evaluate_mini_batch(model, obs_feats)
            train_acc = evaluator(obs_out[obs_idx_train], obs_labels[obs_idx_train])
            val_acc = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
            _, out = evaluate_mini_batch(model, feats)
            test_acc = evaluator(out[idx_test_ind], labels[idx_test_ind])
            
        if epoch % 1 == 0:
            print("\033[0;30;43m [{}] CLA: {:.5f}, KD: {:.5f}, Total: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Power: {:.4f}\033[0m".format(
                                        epoch, loss_l, loss_t, loss_l + loss_t, train_acc, val_acc, test_acc, val_best, test_val, test_best, KD_model.power))

        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            test_val = test_acc
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1
            
        if es == 50:
            print("Early stopping!")
            break

    model.load_state_dict(state)
    model.eval()
    if param['exp_setting'] == 'tran':
        out, _ = evaluate_mini_batch(model, feats)
    else:
        obs_out, _ = evaluate_mini_batch(model, obs_feats)
        out, _ = evaluate_mini_batch(model, feats)
        out[idx_obs] = obs_out

    return out, test_acc, test_val, test_best, state