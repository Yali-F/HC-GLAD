import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from models.model import HC
from utils.data_loader import *
import argparse 
import numpy as np
import torch
import random
import sklearn.metrics as skm
import torch_geometric
from optimizers.radam import RiemannianSGD
import gc



def arg_parse():
    parser = argparse.ArgumentParser(description='HC-GLAD')
    parser.add_argument('-pooling', type=str, default='add', choices=['add', 'max', 'mean'])
    parser.add_argument('-readout', type=str, default='concat', choices=['concat', 'add', 'last'])
    parser.add_argument('-exp_type', type=str, default='ad', choices=['oodd', 'ad'])
    parser.add_argument('-DS', help='Dataset', default='BZR')
    parser.add_argument('-DS_ood', help='Dataset', default='COX2')
    parser.add_argument('-DS_pair', default=None)
    parser.add_argument('-rw_dim', type=int, default=16)      
    parser.add_argument('-dg_dim', type=int, default=16)     
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size_test', type=int, default=9999)
    parser.add_argument('-lr', type=float, default=0.0001)                   
    parser.add_argument('-num_layers', type=int, default=5)  
    parser.add_argument('-hidden_dim', type=int, default=16)   
    parser.add_argument('-num_trial', type=int, default=5)    
    parser.add_argument('-num_epoch', type=int, default=400)   
    parser.add_argument('-eval_freq', type=int, default=10)    
    parser.add_argument('-GNN_Encoder', type=str, default='GCN', choices=['GCN', 'GIN'])                                       
    parser.add_argument('-optimizer', type=str, default='RiemannianSGD', help='which optimizer to use, can be any of [Adam, RiemannianAdam]')
    parser.add_argument('-c', type=float, default=1.0, help='hyperbolic radius, set to None for trainable curvature')
    parser.add_argument('-model', type=str, default='HGCN')
    parser.add_argument('-manifold', type=str, default='Hyperboloid')
    parser.add_argument('-weight_decay', type=float, default=0.005, help='l2 regularization strength')
    parser.add_argument('-momentum', type=float, default=0.95, help='momentum in optimizer')
    parser.add_argument('-adjust_lr', type=int, default=0, help='whether to adjust learning rate')
    parser.add_argument('-lr_reduce_freq', type=int, default=None, help='reduce lr every lr-reduce-freq or None to keep lr constant')
    parser.add_argument('-gamma', type=float, default=0.5, help='gamma for lr scheduler')
    parser.add_argument('-a', type=float, default=0.5)                                            
    parser.add_argument('-b', type=float, default=0.5, help='loss for hyperGraph')
    
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed) 


if __name__ == '__main__': 

    setup_seed(0)
    args = arg_parse()


    if args.exp_type == 'ad':
        if args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_Tox21(args)
        else:
            splits = get_ad_split_TU(args, fold=args.num_trial)
    
    tot_auc_list = [[] for _ in range(args.num_epoch // args.eval_freq)]

    aucs = []
    for trial in range(args.num_trial):
        setup_seed(trial + 1)

        if args.exp_type == 'oodd':
            dataloader, dataloader_test, meta = get_ood_dataset(args)
        elif args.exp_type == 'ad' and not args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_TU(args, splits[trial])

        args.feat_dim = meta['num_feat']
        n_train = meta['num_train']                            
       
        if not args.lr_reduce_freq:
            args.lr_reduce_freq = args.num_epoch

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        model = HC(args).to(device)
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = RiemannianSGD(params=model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, momentum=args.momentum)
        if args.adjust_lr:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(                                                          
                optimizer,
                step_size=int(args.lr_reduce_freq),
                gamma=float(args.gamma)
            )


        for epoch in range(1, args.num_epoch + 1):             
            model.train()
            loss_all = 0

            for data in dataloader:

                gc.collect()
                torch.cuda.empty_cache()

                data = data.to(device)
                optimizer.zero_grad()
 
             
                g_hBol, g_f_hBol, n_f_hBol, g_s_hBol, n_s_hBol, g_hBol_hGra, g_f_hBol_hGra, n_f_hBol_hGra, g_s_hBol_hGra, n_s_hBol_hGra  = model.encode(data)
                

                loss_g_hBol = model.calc_loss_g(g_f_hBol, g_s_hBol)

                loss_n_hBol = model.calc_loss_n(n_f_hBol, n_s_hBol, data.batch)
                

                loss_g_hBol_hGra = model.calc_loss_g(g_f_hBol_hGra, g_s_hBol_hGra)   
                
                loss_n_hBol_hGra = model.calc_loss_n(n_f_hBol_hGra, n_s_hBol_hGra, data.batch)
               
                loss = args.a * (loss_g_hBol.mean() + loss_n_hBol.mean()) + args.b * (loss_g_hBol_hGra.mean() + loss_n_hBol_hGra.mean())


                loss_all += loss.item() * data.num_graphs                  
                loss.backward()
                optimizer.step()
                if args.adjust_lr:
                    lr_scheduler.step()


            print('Current Trial: {:02d}'.format(trial))
            print('[TRAIN] Epoch:{:03d} | Loss:{:.4f}'.format(epoch, loss_all / n_train))

               
            if epoch % args.eval_freq == 0:
                model.eval()

                y_score_all = []
                y_true_all = []
                for data in dataloader_test:
                    with torch.no_grad():
                        gc.collect()
                        torch.cuda.empty_cache()

                        data = data.to(device)

                        g_hBol, g_f_hBol, n_f_hBol, g_s_hBol, n_s_hBol, g_hBol_hGra, g_f_hBol_hGra, n_f_hBol_hGra, g_s_hBol_hGra, n_s_hBol_hGra  = model.encode(data)
                      
                        y_score_g = model.calc_loss_g(g_f_hBol, g_s_hBol)
                      
                        y_score_n = model.calc_loss_n(n_f_hBol, n_s_hBol, data.batch)
                       
                        y_score_g_hyper = model.calc_loss_g(g_f_hBol_hGra, g_s_hBol_hGra)

                        y_score_n_hyper = model.calc_loss_n(n_f_hBol_hGra, n_s_hBol_hGra, data.batch)
                                                
                        y_score = args.a * (y_score_g + y_score_n) + args.b * (y_score_g_hyper + y_score_n_hyper)


                        y_true = data.y

                        y_score_all = y_score_all + y_score.detach().cpu().tolist()
                        y_true_all = y_true_all + y_true.detach().cpu().tolist()

                auc = skm.roc_auc_score(y_true_all, y_score_all)

                print('[EVAL] Epoch: {:03d} | AUC:{:.4f}'.format(epoch, auc))

                tot_auc_list[epoch // args.eval_freq - 1].append(auc)           

        print('[RESULT] Trial: {:02d} | AUC:{:.4f}'.format(trial, auc))
        aucs.append(auc)


    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    # print('[FINAL RESULT] AVG_AUC:{:.4f}+-{:.4f}'.format(avg_auc, std_auc))

    auc_list = [(np.mean(auc), np.std(auc), (idx + 1) * args.eval_freq) for idx, auc in enumerate(tot_auc_list)]
    for row in auc_list:
        print(row)
    auc_list.sort(key = lambda x: (-x[0], x[1], x[2]))

    print(args)
    print('[The Best result is] Avg_Auc:{:.4f} +- {:.4f}, achieved in {} epoch'.format(auc_list[0][0], auc_list[0][1], auc_list[0][2]) )
    print(args.exp_type, args.DS)