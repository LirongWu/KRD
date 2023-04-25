import csv
import nni
import time
import json
import argparse
import warnings
import numpy as np
import torch
import torch.optim as optim

from utils import *
from models import *
from dataloader import *
from train_and_eval import *

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():

    g, labels, idx_train, idx_val, idx_test = load_data(param['dataset'], seed=param['seed'])
    feats = g.ndata["feat"].to(device)
    labels = labels.to(device)
    param['feat_dim'] = g.ndata["feat"].shape[1]
    param['label_dim'] = labels.int().max().item() + 1

    if param['exp_setting'] == "tran":
        output_dir = Path.cwd().joinpath("../outputs", "transductive", param['dataset'], f"{param['teacher']}_{param['student']}", f"seed_{param['seed']}")
        indices = (idx_train, idx_val, idx_test)
    elif param['exp_setting'] == "ind":
        output_dir = Path.cwd().joinpath("../outputs", "inductive", param['dataset'], f"{param['teacher']}_{param['student']}", f"seed_{param['seed']}")
        indices = graph_split(idx_train, idx_val, idx_test, labels, param)
    else:
        raise ValueError(f"Unknown experiment setting! {param['exp_setting']}")
    check_writable(output_dir, overwrite=False)


    criterion_l = torch.nn.NLLLoss()
    criterion_t = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    evaluator = get_evaluator(param["dataset"])


    if param['ablation_mode'] == 0:

        model_t = Model(param, model_type='teacher').to(device)
        optimizer_t = optim.Adam(model_t.parameters(), lr=float(1e-2), weight_decay=float(param["weight_decay"]))
        out_t, _, test_teacher, _, _ = train_teacher(param, model_t, g, feats, labels, indices, criterion_l, evaluator, optimizer_t)

        KD_model = Com_KD_Prob(param)
        if param['exp_setting'] == 'tran':
            KD_model.initialization(model_t, g, feats)
        else:
            KD_model.initialization(model_t, g.subgraph(indices[3]), feats[indices[3]])

        model_s = Model(param, model_type='student').to(device)
        optimizer_s = optim.Adam(model_s.parameters(), lr=float(param["learning_rate"]), weight_decay=float(param["weight_decay"]))
        _, test_acc, test_val, test_best, _ = train_student(param, model_s, g, feats, labels, out_t, indices, criterion_l, criterion_t, evaluator, optimizer_s, KD_model) 
        
        return test_teacher, test_acc, test_val, test_best


    elif param['ablation_mode'] == 1:

        model_t = Model(param, model_type='teacher').to(device)
        optimizer_t = optim.Adam(model_t.parameters(), lr=float(1e-2), weight_decay=float(param["weight_decay"]))
        out_t, test_acc, test_val, test_best, state_t = train_teacher(param, model_t, g, feats, labels, indices, criterion_l, evaluator, optimizer_t)

        np.savez(output_dir.joinpath("out_teacher"), out_t.detach().cpu().numpy())
        torch.save(state_t, output_dir.joinpath("model_teacher"))

        return test_val, test_acc, test_val, test_best


    elif param['ablation_mode'] == 2:

        out_t = load_out_t(output_dir).to(device)
        model_t = Model(param, model_type='teacher').to(device)
        state_t = torch.load(output_dir.joinpath("model_teacher"))
        model_t.load_state_dict(state_t)

        KD_model = Com_KD_Prob(param)
        if param['exp_setting'] == 'tran':
            KD_model.initialization(model_t, g, feats)
            test_teacher = evaluator(out_t[indices[2]].log_softmax(dim=1), labels[indices[2]])
        else:
            KD_model.initialization(model_t, g.subgraph(indices[3]), feats[indices[3]])
            test_teacher = evaluator(out_t[indices[4]].log_softmax(dim=1), labels[indices[4]])
        
        model_s = Model(param, model_type='student').to(device)
        optimizer_s = optim.Adam(model_s.parameters(), lr=float(param["learning_rate"]), weight_decay=float(param["weight_decay"]))
        out_s, test_acc, test_val, test_best, state_s = train_student(param, model_s, g, feats, labels, out_t, indices, criterion_l, criterion_t, evaluator, optimizer_s, KD_model)  

        np.savez(output_dir.joinpath("out_student"), out_s.detach().cpu().numpy())
        torch.save(state_s, output_dir.joinpath("model_student"))

        return test_teacher, test_acc, test_val, test_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--teacher", type=str, default="GCN")
    parser.add_argument("--student", type=str, default="MLP")
    parser.add_argument("--exp_setting", type=int, default=0)
    parser.add_argument("--split_rate", type=float, default=0.1)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout_t", type=float, default=0.8)
    parser.add_argument("--dropout_s", type=float, default=0.6)

    parser.add_argument("--lamb", type=float,default=0.0)
    parser.add_argument("--tau", type=float,default=1.0)
    parser.add_argument("--init_power", type=float,default=1.0)
    parser.add_argument("--momentum", type=float,default=0.9)
    parser.add_argument("--bins_num", type=float,default=50)

    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--max_epoch", type=int, default=500)
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_mode", type=int, default=0)
    parser.add_argument("--data_mode", type=int, default=0)
    parser.add_argument("--ablation_mode", type=int, default=0)

    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())

    if param['data_mode'] == 0:
        param['dataset'] = 'cora'
    if param['data_mode'] == 1:
        param['dataset'] = 'citeseer'
    if param['data_mode'] == 2:
        param['dataset'] = 'pubmed'
    if param['data_mode'] == 3:
        param['dataset'] = 'coauthor-cs'
    if param['data_mode'] == 4:
        param['dataset'] = 'coauthor-phy'
    if param['data_mode'] == 5:
        param['dataset'] = 'amazon-photo'
    if param['data_mode'] == 6:
        param['dataset'] = 'amazon-com'
    if param['data_mode'] == 7:
        param['dataset'] = 'ogbn-arxiv'

    if param['data_mode'] == 7:
        param['norm_type'] = 'batch'
    else:
        param['norm_type'] = 'none'
    if param['exp_setting'] == 0:
        param['exp_setting'] = 'tran'
    else:
        param['exp_setting'] = 'ind'

    if os.path.exists("../param/best_parameters.json"):
        param = json.loads(open("../param/best_parameters.json", 'r').read())[param['dataset']][param['exp_setting']][param['teacher']]

    if param['save_mode'] == 0:
        set_seed(param['seed'])
        test_teacher, test_acc, test_val, test_best = main()
        nni.report_final_result(test_val)

    else:
        test_acc_list = []
        test_val_list = []
        test_best_list = []
        test_teacher_list = []

        for seed in range(5):
            param['seed'] += seed
            set_seed(param['seed'])
            test_teacher, test_acc, test_val, test_best = main()
            
            test_acc_list.append(test_acc)
            test_val_list.append(test_val)
            test_best_list.append(test_best)
            test_teacher_list.append(test_teacher)
            nni.report_intermediate_result(test_val)
            
        nni.report_final_result(np.mean(test_val_list))

    outFile = open('../PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    for v, k in param.items():
        results.append(k)
    
    if param['save_mode'] == 0:
        results.append(str(test_acc))
        results.append(str(test_val))
        results.append(str(test_best))
        results.append(str(test_teacher))

    else:  
        results.append(str(test_acc_list))
        results.append(str(test_val_list))
        results.append(str(test_best_list))
        results.append(str(test_teacher_list))
        results.append(str(np.mean(test_acc_list)))
        results.append(str(np.mean(test_val_list)))
        results.append(str(np.mean(test_best_list)))
        results.append(str(np.mean(test_teacher_list)))
        results.append(str(np.std(test_acc_list)))
        results.append(str(np.std(test_val_list)))
        results.append(str(np.std(test_best_list)))
        results.append(str(np.std(test_teacher_list)))
    writer.writerow(results)