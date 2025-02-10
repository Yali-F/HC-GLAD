
import torch.nn as nn
import manifolds
import layers.hyp_layers as hyp_layers
import utils.math_utils as pmath


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, data):

        input = (data, False)
        hypergraph_input = (data, True)
           
        hyperbolic_g, hyperbolic_g_f, hyperbolic_n_f, hyperbolic_g_s, hyperbolic_n_s = self.layers.forward(input) 
        hyperbolic_hypergraph_g, hyperbolic_hypergraph_g_f, hyperbolic_hypergraph_n_f, hyperbolic_hypergraph_g_s, hyperbolic_hypergraph_n_s = self.hypergraph_layers.forward(hypergraph_input)
                        
        return hyperbolic_g, hyperbolic_g_f, hyperbolic_n_f, hyperbolic_g_s, hyperbolic_n_s, hyperbolic_hypergraph_g, hyperbolic_hypergraph_g_f, hyperbolic_hypergraph_n_f, hyperbolic_hypergraph_g_s, hyperbolic_hypergraph_n_s        



class HGCN(Encoder):                                                                                      
    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1

        hgc_layers = []
        hypergraph_hgc_layers = []


        in_dim_f = args.feat_dim + 1                                                                     
        in_dim_s = args.rw_dim + args.dg_dim + 1        

        out_dim = args.hidden_dim

        hgc_layers.append(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, in_dim_f, in_dim_s, out_dim, self.c, args.num_layers, args, is_hypergraph = False
            )
        )
        hypergraph_hgc_layers.append(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, in_dim_f, in_dim_s, out_dim, self.c, args.num_layers, args, is_hypergraph = True
            )
        )
        
        self.layers = nn.Sequential(*hgc_layers)#.to(args.device)                                      
        self.hypergraph_layers = nn.Sequential(*hypergraph_hgc_layers)


    def encode(self, data):
        data.x = self.manifold.proj(data.x, c=self.c)
        data.x_s = self.manifold.proj(data.x_s, c=self.c)

        return super(HGCN, self).encode(data)                                                            
