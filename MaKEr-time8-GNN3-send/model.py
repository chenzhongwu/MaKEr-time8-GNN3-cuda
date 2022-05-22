import torch.nn as nn
import torch
import dgl
from ext_gnn import ExtGNN


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.dim = args.dim
        # 因为模式用的是ROTATE，所以实体维度是关系的两倍。

        # 每一次args.num_rel、args.num_ent都是不一样的？
        # self.rel_comp：154*4，一共154种关系，用于处初始化RPG中关系的初始表达。
        self.rel_comp = nn.Parameter(torch.Tensor(args.num_rel, args.num_rel_bases))
        nn.init.xavier_uniform_(self.rel_comp, gain=nn.init.calculate_gain('relu'))
        # self.rel_feat： 4*32
        self.rel_feat = nn.Parameter(torch.Tensor(args.num_rel_bases, self.args.rel_dim))
        nn.init.xavier_uniform_(self.rel_feat, gain=nn.init.calculate_gain('relu'))
        # self.ent_feat ： 952*64   实体表达                                 对于可见的实体/关系，分别单独学习一种emb。
        self.ent_feat = nn.Parameter(torch.Tensor(args.num_ent, self.args.ent_dim))
        nn.init.xavier_uniform_(self.ent_feat, gain=nn.init.calculate_gain('relu'))
        # self.rel_head_feat：4*64
        self.rel_head_feat = nn.Parameter(torch.Tensor(args.num_rel_bases, self.args.ent_dim))
        self.rel_tail_feat = nn.Parameter(torch.Tensor(args.num_rel_bases, self.args.ent_dim))
        nn.init.xavier_uniform_(self.rel_head_feat, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_tail_feat, gain=nn.init.calculate_gain('relu'))

        # 4种元关系：4*4 ，即tt、th、hh、ht    for initializing relation in pattern graph (relation position graph)
        self.pattern_rel_ent = nn.Parameter(torch.Tensor(4, args.num_rel_bases))
        nn.init.xavier_uniform_(self.pattern_rel_ent, gain=nn.init.calculate_gain('relu'))
        # 最后一步外推
        self.ext_gnn = ExtGNN(args)

        # 我们加入绝对时间
        self.time_feat = nn.Parameter(torch.Tensor(args.num_time, self.args.time_dim))
        nn.init.xavier_uniform_(self.time_feat, gain=nn.init.calculate_gain('relu'))

    # relation feature representation
    def init_pattern_g(self, pattern_g):
        with pattern_g.local_scope():
            # 关系部分
            e_rel_types = pattern_g.edata['rel']
            pattern_g.edata['edge_rel'] = self.pattern_rel_ent[e_rel_types]

            message_func = dgl.function.copy_e('edge_rel', 'msg')
            reduce_func = dgl.function.mean('msg', 'RPG_rel')
            pattern_g.update_all(message_func, reduce_func)
            pattern_g.edata.pop('edge_rel')

            # 时间部分，RPG上时间不太好加，先不加
            # e_time_types = pattern_g.edata['time']
            # pattern_g.edata['edge_time'] = self.pattern_rel_ent[e_time_types]
            # message_func = dgl.function.copy_e('edge_time', 'msg')
            # reduce_func = dgl.function.mean('msg', 'RPG_time')
            # pattern_g.update_all(message_func, reduce_func)
            # pattern_g.edata.pop('edge_time')

            # for observed rel
            # 对于见过的关系，  是有对应的关系类型的，他们的表达直接用初始化的对应关系类型的表达。
            # 对于没见过的关系，是通过上面聚合4种模式得到表达，不用替换。
            obs_idx = (pattern_g.ndata['ori_idx'] != -1)
            pattern_g.ndata['RPG_rel'][obs_idx] = self.rel_comp[pattern_g.ndata['ori_idx'][obs_idx]]

            rel_coef = pattern_g.ndata['RPG_rel']
            # time_coef = pattern_g.ndata['RPG_time']
        # rel_coef就是seen和unseen的关系在RPG中的表示
        # return rel_coef, time_coef
        return rel_coef

    # entity feature representation
    def init_g(self, g, rel_coef):
        with g.local_scope():
            num_edge = g.num_edges()
            etypes = g.edata['b_rel']
            # 一个是inW，一个是outW。公式4的求和部分，先求和好，等待取用
            # 将4维的元关系通过公式4中的W转换为64维
            rel_head_emb = torch.matmul(rel_coef, self.rel_head_feat)
            rel_tail_emb = torch.matmul(rel_coef, self.rel_tail_feat)

            g.edata['edge_h'] = torch.zeros(num_edge, self.args.ent_dim).to(self.args.gpu)

            non_inv_idx = (g.edata['inv'] == 0)
            inv_idx = (g.edata['inv'] == 1)
            g.edata['edge_h'][inv_idx] = rel_head_emb[etypes[inv_idx]]
            g.edata['edge_h'][non_inv_idx] = rel_tail_emb[etypes[non_inv_idx]]
            # 执行求和
            message_func = dgl.function.copy_e('edge_h', 'msg')
            reduce_func = dgl.function.mean('msg', 'h')
            g.update_all(message_func, reduce_func)
            g.edata.pop('edge_h')

            # 这里加入时间也应该是加入的一种模式，而不能是绝对的时间戳
            # for observed ent 换掉可见的
            obs_idx = (g.ndata['ori_idx'] != -1)
            g.ndata['h'][obs_idx] = self.ent_feat[g.ndata['ori_idx'][obs_idx]]

            ent_feat = g.ndata['h']

        return ent_feat

    # self.model(batch_sup_g, batch_pattern_g) 实体图+RPG
    def forward(self, g, pattern_g):
        rel_coef = self.init_pattern_g(pattern_g)
        # rel_coef即是（c）到（d）的那条线，传入关系信息，就是RPG中用元关系更新后的关系。
        init_ent_feat = self.init_g(g, rel_coef)
        # 将4维的关系表达（因为在RPG中，定义的4种元关系的表达维度也是4，所以导致关系的维度是4）
        # 这里相当于维度扩充，由4扩充到32
        init_rel_feat = torch.matmul(rel_coef, self.rel_feat)
        # init_time_feat =

        # 因为我们目前使用的是绝对时间，所以在GNN更新的时候，直接聚合时间
        ent_emb, rel_emb, time_emb = self.ext_gnn(g, ent_feat=init_ent_feat, rel_feat=init_rel_feat, time_feat = self.time_feat)

        return ent_emb, rel_emb, time_emb