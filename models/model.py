from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn
import manifolds
import models.encoders as encoders




class HC(nn.Module):         
    def __init__(self, args):
        super(HC, self).__init__()
         
        self.embedding_dim = args.hidden_dim   
        if args.readout == 'concat':
            self.embedding_dim *= args.num_layers

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.c = torch.tensor([args.c]).to(device)
        self.manifold = getattr(manifolds, args.manifold)() 
        self.encoder = getattr(encoders, args.model)(self.c, args)#.to(args.device)
        self.num_layers = args.num_layers


    def encode(self, data):    
        o = torch.zeros_like(data.x)      
        data.x = torch.cat([o[:, 0:1], data.x], dim=1)                                                              
        data.x_s = torch.cat([o[:, 0:1], data.x_s], dim=1)                                                     

        data.x = self.manifold.expmap0(data.x, self.c)
        data.x_s = self.manifold.expmap0(data.x_s, self.c)

        hBol_g, hBol_g_f, hBol_n_f, hBol_g_s, hBol_n_s, hBol_hGra_g, hBol_hGra_g_f, hBol_hGra_n_f, hBol_hGra_g_s, hBol_hGra_n_s = self.encoder.encode(data)

        return hBol_g, hBol_g_f, hBol_n_f, hBol_g_s, hBol_n_s, hBol_hGra_g, hBol_hGra_g_f, hBol_hGra_n_f, hBol_hGra_g_s, hBol_hGra_n_s
    

    def calc_loss_n(self, x, x_aug, batch, temperature=1.0):
        batch_size, _ = x.size()                                      
        
        x_abs = x.norm(dim=1)              

        x_aug_abs = x_aug.norm(dim=1)

        node_belonging_mask = batch.repeat(batch_size,1)
        node_belonging_mask = node_belonging_mask == node_belonging_mask.t()
       
        dist_matrix = self.manifold.dist(x.unsqueeze(1), x_aug.unsqueeze(0), self.c)
        dist_matrix = torch.squeeze(dist_matrix, dim=2)
        sim_matrix = dist_matrix
        
        sim_matrix = torch.exp(-sim_matrix / temperature) * node_belonging_mask     

        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-12)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-12)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        loss = global_mean_pool(loss, batch)

        return loss


    def calc_loss_g(self, x, x_aug, temperature=1.0):
        batch_size, _ = x.size()   

        dist_matrix = self.manifold.dist(x.unsqueeze(1), x_aug.unsqueeze(0), self.c)
        dist_matrix = torch.squeeze(dist_matrix, dim=2)
        sim_matrix = dist_matrix

        sim_matrix = torch.exp(-sim_matrix / temperature)

        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        
        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        return loss