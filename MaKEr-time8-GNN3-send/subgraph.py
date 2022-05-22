import pickle
import numpy as np
from utils import get_g, serialize
import torch
import lmdb
import dgl
from collections import defaultdict as ddict
from tqdm import tqdm
import random
from scipy import sparse
import multiprocessing as mp
# from random_icews14 import build_icews_data

# 本文件的任务是 sample10000个子任务
# https://www.cnblogs.com/zhangxianrong/p/14919706.html
# lmbd数据库介绍

def gen_subgraph_datasets(args):
    print('----------generate tasks(sub-KGs) for meta-training----------')
    # data = pickle.load(open(args.data_path, 'rb'))
    # icews_data = build_icews_data()
    with open("./icews14.pickle", "rb") as fp:  # Pickling
        icews_data = pickle.load(fp)

    # 解构data：data是本文新提出的数据集，没有给出创建数据集的过程
    # data[train][ent2id]训练集的实体id
    #            [rel2id]训练集的关系id
    #            [triples]训练集的（h，r，t）

    # data[test][ent2id]测试集的实体id，从0开始独立于训练集标号
    # data[test][ent_map_list]指示每个实体是否在训练集中出现过，-1为未出现，其他为实体在训练集中的位置
    # data[test][support]测试集中所有看见过的triple为支持集
    # data[test][query]分为3部分实体未见过、关系未见过、都未见过

    # data[test][query]只有一种
    # 我们只将包含至少一个看不见的组件的三元组放入测试KG的查询三元组中。 我们还将查询三元组划分为仅包含不可见实体(uent)、只包含不可见关系(urel) 和同时包含不可见实体和不可见关系(uboth) 的三元组。
    # 为了为每个 sub-KG 制定一个任务，我们将三元组的一部分拆分为查询三元组，并将剩余的三元组视为支持三元组。支持三元组用于生成实体嵌入，查询三元组用于评估生成的嵌入的合理性并计算训练损失。


    # 为了加入时间:
    # train还要加time2id,
    # test 还要加time2id,time_map_list,query_utime
    # valid还要加time2id,time_map_list

    # 训练集的4元组
    bg_train_g = get_g(icews_data['train']['quater'])

    # 下两行是计算  新数据库所需磁盘空间的最小值
    BYTES_PER_DATUM = get_average_subgraph_size(args, args.num_sample_for_estimate_size, bg_train_g) * 20
    map_size = (args.num_train_subgraph) * BYTES_PER_DATUM
    # lmdb.open：创建 lmdb 环境，会在指定路径下创建 data.mdb 和 lock.mdb 两个文件，一是个数据文件，一个是锁文件。
    env = lmdb.open(args.db_path, map_size=map_size, max_dbs=1)
    # 文件名称
    train_subgraphs_db = env.open_db("train_subgraphs".encode())

    with mp.Pool(processes=10, initializer=intialize_worker, initargs=(args, bg_train_g)) as p:
        # 生成10000个子任务
        idx_ = range(args.num_train_subgraph)
        # pool将函数sample_one_subgraph分配到进程，函数的输出结果是(str_id, datum)
        for (str_id, datum) in tqdm(p.imap(sample_one_subgraph, idx_), total=args.num_train_subgraph):
            # 参数write设置为True才可以写入
            with env.begin(write=True, db=train_subgraphs_db) as txn:
                # 添加数据和键值 key和value
                # txn是写入和读出的操作对象
                txn.put(str_id, serialize(datum))


def intialize_worker(args, bg_train_g):
    global args_, bg_train_g_
    args_, bg_train_g_ = args, bg_train_g


def sample_one_subgraph(idx_):
    args = args_
    bg_train_g = bg_train_g_

    # get graph with bi-direction
    # 获取头尾实体
    bg_train_g_undir = dgl.graph(( torch.cat([bg_train_g.edges()[0], bg_train_g.edges()[1]]),
                                   torch.cat([bg_train_g.edges()[1], bg_train_g.edges()[0]])  ))

    # induce sub-graph by sampled nodes
    while True:
        while True:
            sel_nodes = []
            for i in range(args.rw_0):
                if i == 0:
                    cand_nodes = np.arange(bg_train_g.num_nodes())
                else:
                    cand_nodes = sel_nodes
                try:
                    # https://docs.dgl.ai/generated/dgl.sampling.random_walk.html
                    rw, _ = dgl.sampling.random_walk(bg_train_g_undir,
                                                     np.random.choice(cand_nodes, 1, replace=False).repeat(args.rw_1),
                                                     length=args.rw_2)
                except ValueError:
                    print(cand_nodes)
                # print(rw)
                sel_nodes.extend(np.unique(rw.reshape(-1)))
                # -1是走到头的边
                sel_nodes = list(np.unique(sel_nodes)) if -1 not in sel_nodes else list(np.unique(sel_nodes))[1:]

            # dgl.node_subgraph : Return a subgraph induced on the given nodes.
            # In addition to extracting the subgraph, DGL conducts the following:
            # Relabel the extracted nodes to IDs starting from zero.
            # Copy the features of the extracted nodes and edges to the resulting graph. The copy is lazy and incurs data movement only when needed.
            sub_g = dgl.node_subgraph(bg_train_g, sel_nodes)

            if sub_g.num_nodes() >= 50:
                break

        # 文章sample出seen和unseen节点都从0开始编号，相当于一个异构图，有两种类型的节点。
        # https://docs.dgl.ai/en/0.6.x/guide_cn/graph-heterogeneous.html#guide-cn-graph-heterogeneous
        # sub_g.ndata[dgl.NID] ： 原始的特定类型节点ID
        sub_tri = torch.stack([sub_g.ndata[dgl.NID][sub_g.edges()[0]],
                               sub_g.edata['rel'],
                               sub_g.ndata[dgl.NID][sub_g.edges()[1]],
                               sub_g.edata['time']
                               ])

        sub_tri = sub_tri.T.tolist()

        random.shuffle(sub_tri)

        ent_freq = ddict(int)
        rel_freq = ddict(int)
        time_freq = ddict(int)
        triples_reidx = []

        rel_reidx = dict()
        relidx = 0

        ent_reidx = dict()
        entidx = 0

        time_reidx = dict()
        timeidx = 0

        for tri in sub_tri:
            h, r, t, time  = tri
            # h, r, t  = tri
            if h not in ent_reidx.keys():
                ent_reidx[h] = entidx
                entidx += 1
            if t not in ent_reidx.keys():
                ent_reidx[t] = entidx
                entidx += 1
            if r not in rel_reidx.keys():
                rel_reidx[r] = relidx
                relidx += 1
            if time not in time_reidx.keys():
                time_reidx[time] = timeidx
                timeidx += 1

            ent_freq[ent_reidx[h]] += 1
            ent_freq[ent_reidx[t]] += 1
            rel_freq[rel_reidx[r]] += 1
            time_freq[time_reidx[time]] += 1
            triples_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t], time_reidx[time]])
            # triples_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])

        ent_reidx_inv = {v: k for k, v in ent_reidx.items()}
        rel_reidx_inv = {v: k for k, v in rel_reidx.items()}
        time_reidx_inv = {v: k for k, v in time_reidx.items()}
        ent_map_list = [ent_reidx_inv[i] for i in range(len(ent_reidx))]
        rel_map_list = [rel_reidx_inv[i] for i in range(len(rel_reidx))]
        time_map_list = [time_reidx_inv[i] for i in range(len(time_reidx))]

        # randomly get query triples
        # 生成query和support
        que_tris = []
        sup_tris = []
        for idx, tri in enumerate(triples_reidx):
            # h, r, t = tri
            h, r, t, time = tri
            if ent_freq[h] > 2 and ent_freq[t] > 2 and rel_freq[r] > 2 and time_freq[time] > 2:
                que_tris.append(tri)
                ent_freq[h] -= 1
                ent_freq[t] -= 1
                rel_freq[r] -= 1
                time_freq[time] -= 1
            else:
                sup_tris.append(tri)

            if len(que_tris) >= int(len(triples_reidx)*0.1):
                break

        sup_tris.extend(triples_reidx[idx+1:])
        # que_tris数量够了，退出大循环
        if len(que_tris) >= int(len(triples_reidx)*0.1):
            break

    # hr2t, rt2h
    hr2t, rt2h, rel_head, rel_tail = get_hr2t_rt2h_sup_que(sup_tris, que_tris)
    pattern_tris = get_train_pattern_g(rel_head, rel_tail)

    str_id = '{:08}'.format(idx_).encode('utf-8')

    return str_id, (sup_tris, pattern_tris, que_tris, hr2t, rt2h, ent_map_list, rel_map_list)


def get_train_pattern_g(rel_head, rel_tail):
    # adjacency matrix for rel and rel of different pattern
    # 一乘就是4种元关系下，每种关系之间对应三元关系
    tail_head = torch.matmul(rel_tail, rel_head.T)
    head_tail = torch.matmul(rel_head, rel_tail.T)
    tail_tail = torch.matmul(rel_tail, rel_tail.T) - torch.diag(torch.sum(rel_tail, axis=1))
    head_head = torch.matmul(rel_head, rel_head.T) - torch.diag(torch.sum(rel_head, axis=1))

    # construct pattern graph from adjacency matrix
    src = torch.LongTensor([])
    dst = torch.LongTensor([])
    p_rel = torch.LongTensor([])
    p_w = torch.LongTensor([])
    # 将4种元关系矩阵转换为链接关系，cat在一起，因为输出是用来构造图的，不用区分元关系0到3
    for p_rel_idx, mat in enumerate([tail_head, head_tail, tail_tail, head_head]):
        sp_mat = sparse.coo_matrix(mat)
        src = torch.cat([src, torch.from_numpy(sp_mat.row)])
        dst = torch.cat([dst, torch.from_numpy(sp_mat.col)])
        p_rel = torch.cat([p_rel, torch.LongTensor([p_rel_idx] * len(sp_mat.data))])
        p_w = torch.cat([p_w, torch.from_numpy(sp_mat.data)])

    return torch.stack([src, p_rel, dst]).T.tolist()


def get_average_subgraph_size(args, sample_size, bg_train_g):
    total_size = 0

    with mp.Pool(processes=10, initializer=intialize_worker, initargs=(args, bg_train_g)) as p:
        idx_ = range(sample_size)
        for (str_id, datum) in p.imap(sample_one_subgraph, idx_):
            total_size += len(serialize(datum))

    return total_size / sample_size


def get_hr2t_rt2h_sup_que(sup_tris, que_tris):
    hrtime2t = ddict(list)
    rttime2h = ddict(list)

    triples = torch.LongTensor(sup_tris)
    num_rel = torch.unique(triples[:, 1]).shape[0]
    num_ent = torch.unique(torch.cat((triples[:, 0], triples[:, 2]))).shape[0]
    num_time = torch.unique(triples[:, 3]).shape[0]
    # num_time = torch.unique(triples[:, 3]).shape[0]
    # 问题就在这，怎么加时间，如果是使用相对时间，这里要再学习一种时间的模式。
    # 目前先学习绝对时间，将时间embedding化。在GNN的结构和ConvE的得分函数里加进去
    rel_head = torch.zeros((num_rel, num_ent), dtype=torch.int)
    rel_tail = torch.zeros((num_rel, num_ent), dtype=torch.int)

    for tri in sup_tris:
        h, r, t,time = tri
        hrtime2t[(h, r, time)].append(t)
        rttime2h[(r, t ,time)].append(h)
        # rel_head统计每个[r, h]出现的次数，实际上是记录该rel 是否 以该实体为head
        # 构造RPG只在sup上构建
        rel_head[r, h] += 1
        rel_tail[r, t] += 1
    # hr2t由sup和que统一维护，记录所有unseen的三元组。
    for tri in que_tris:
        h, r, t, time= tri
        hrtime2t[(h, r, time)].append(t)
        rttime2h[(r, t, time)].append(h)
    # que直接从hr2t复制过来，得到query三元组
    que_hr2t = dict()
    que_rt2h = dict()
    for tri in que_tris:
        h, r, t, time= tri
        que_hr2t[(h, r, time)] = hrtime2t[(h, r, time)]
        que_rt2h[(r, t, time)] = rttime2h[(r, t, time)]

    return que_hr2t, que_rt2h, rel_head, rel_tail