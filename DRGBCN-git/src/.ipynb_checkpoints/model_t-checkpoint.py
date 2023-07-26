from .model_help import BaseModel
from .dataset import PairGraphData
from . import MODEL_REGISTRY
from .bgnn import GCNConv as GCN

import torch.nn.functional as F
import torch

from torch import nn, optim
from torch.nn.utils.weight_norm import weight_norm


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.4, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k , h_out), dim=None)
        # # Define the bilinear conv layer
        # self.bilinear_conv = nn.Conv2d(h_dim*self.k, h_out, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            # d_ = self.bilinear_conv(v_ * q_)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        # logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        # for i in range(1, self.h_out):
        #     logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
        #     logits += logits_i
        # logits = self.bn(logits)
        return  att_maps


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/bc.py
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim;
        self.q_dim = q_dim
        self.h_dim = h_dim;
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits

        # low-rank bilinear pooling using einsum
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)  # b x v x d
        q_ = self.q_net(q)  # b x q x d
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits

# class MolecularGCN(nn.Module):
#     def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
#         super(MolecularGCN, self).__init__()
#         self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
#         if padding:
#             with torch.no_grad():
#                 self.init_transform.weight[-1].fill_(0)
#         self.gnn = GCN(in_channels=dim_embedding, out_channels=dim_embedding)
#         # self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
#         # self.output_feats = hidden_feats[-1]

#     def forward(self, embedding, edge):
#         if not hasattr(self, "edge_index"):
#             edge_index = torch.sparse_coo_tensor(*edge)
#             self.register_buffer("edge_index", edge_index)
#         edge_index = self.edge_index
#         embedding = self.init_transform(embedding)
#         embedding = self.gnn(embedding, edge_index=edge_index)
#         return embedding


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None, num_layers=3):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        self.att = nn.Parameter(torch.Tensor([0.5, 0.33, 0.25]))
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn_layers = nn.ModuleList([GCN(in_channels=dim_embedding, out_channels=dim_embedding) for i in range(num_layers)])
        # self.output_feats = dim_embedding

    def forward(self, embedding, edge):
        if not hasattr(self, "edge_index"):
            edge_index = torch.sparse_coo_tensor(*edge)
            self.register_buffer("edge_index", edge_index)
        edge_index = self.edge_index
        embedding = self.init_transform(embedding)
        for i, layer in enumerate(self.gnn_layers):
            out = layer(embedding, edge_index=edge_index)
            out = out * self.att[i]
            embedding = embedding + out  # Add residual connection
        return embedding





class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        # x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        x = self.out(x)
        return x





@MODEL_REGISTRY.register()
class Tmodel(BaseModel):
    DATASET_TYPE = "PairGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Tmodel model config")
        parser.add_argument("--embedding_dim", default=64, type=int, help="编码器关联嵌入特征维度")
        parser.add_argument("--hidden_dims", type=int, default=(48, 32), nargs="+", help="解码器每层隐藏单元数")
        parser.add_argument("--lr", type=float, default=2e-3)
        parser.add_argument("--pos_weight", type=float, default=1.0, help="no used, overwrited, use for bce loss")
        parser.add_argument("--loss_fn", type=str, default="bce", choices=["bce", "focal"])
        return parent_parser

    def __init__(self, n_drug, n_disease, embedding_dim=64, hidden_dims=(64, 32),
                 lr=5e-4,  pos_weight=1.0,  loss_fn="bce",  **config):

        super(Tmodel, self).__init__()
        self.n_drug = n_drug
        self.n_disease = n_disease
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims

        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        # "rank bce mse focal"
        self.loss_fn_name = loss_fn
        self.lr = lr
        self.drug_encoder = MolecularGCN(n_drug, embedding_dim)
        self.disease_encoder = MolecularGCN(n_disease, embedding_dim)

        self.bcn = BANLayer(v_dim=embedding_dim, q_dim=embedding_dim, h_dim=embedding_dim, h_out=embedding_dim)
        self.mlp_classifier = MLPDecoder(embedding_dim, hidden_dims[0], hidden_dims[1], binary=1)



    def loss_fn(self, predict, label,u_edge, v_edge):
        bce_loss = self.bce_loss_fn(predict, label, self.pos_weight)
        mse_loss = self.mse_loss_fn(predict, label, self.pos_weight)
        rank_loss = self.rank_loss_fn(predict, label)

        loss = {}
        loss.update(bce_loss)
        loss.update(mse_loss)
        loss.update(rank_loss)
        loss["loss"] = loss[f"loss_{self.loss_fn_name}"]

        return loss

    def step(self, batch:PairGraphData):
        interaction_pairs = batch.interaction_pair
        label = batch.label
        drug_edge = batch.u_edge
        disease_edge = batch.v_edge
        drug_embedding = batch.u_embedding
        disease_embedding = batch.v_embedding

        predict = self.forward(interaction_pairs, drug_edge, disease_edge, drug_embedding, disease_embedding)
        if not self.training:
            predict = predict[batch.valid_mask.reshape(*predict.shape)]
            label = label[batch.valid_mask]
        ans = self.loss_fn(predict=predict, label=label, u_edge=drug_edge, v_edge=disease_edge)
        ans["predict"] = predict
        ans["label"] = label
        return ans

    def training_step(self, batch, batch_idx=None):
        return self.step(batch)

    def validation_step(self, batch, batch_idx=None):
        return self.step(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(lr=self.lr, params=self.parameters(), weight_decay=8e-3)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.05*self.lr, max_lr=self.lr,
                                                   gamma=0.95, mode="exp_range", step_size_up=4,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]

    def forward(self, interaction_pairs, drug_edge, disease_edge, drug_embedding, disease_embedding):

        v_d = self.drug_encoder(drug_embedding, drug_edge)
        # BxTxD
        v_p = self.disease_encoder(disease_embedding, disease_edge)
        # BxMxD
        att = self.bcn(v_d.unsqueeze(0), v_p.unsqueeze(0))
        # BxDxTxM
        B, D, T, M = att.shape
        att = att.permute(0, 2, 3, 1).reshape(-1, D)
        score = self.mlp_classifier(att).reshape(T, M)
        return score[interaction_pairs[0], interaction_pairs[1]]


