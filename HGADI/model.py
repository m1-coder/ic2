import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score,average_precision_score
from torch_geometric.utils import negative_sampling
from torch_sparse import SparseTensor
from collections import defaultdict
import os
from pytorchtools import EarlyStopping
from torch.optim.lr_scheduler import StepLR

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def build_left_right_to_edge_type(edge_type_to_index):

    left_right_to_edge_type = {}
    for (left_type, rel, right_type), edge_type in edge_type_to_index.items():
        key = (left_type, right_type)
        left_right_to_edge_type[key] = [edge_type // 2]# DBLP should set to [edge_type % 3]

    return left_right_to_edge_type

def load_fairwalk_embeddings(path, node_types, node_type_ranges_homo,homo_node_types,embedding_dim=128):
    from collections import defaultdict
    embeddings_dict = {}
    type_to_nodes = defaultdict(list)

    with open(path, 'r') as f:
        lines=f.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            node_id = int(parts[0])
            embedding = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32,device=device)
            if embedding.size(0) != embedding_dim:
                raise ValueError
            embeddings_dict[node_id] = embedding

            node_type1 = node_types[node_id].item()
            node_idtype = homo_node_types[node_type1]
            type_to_nodes[node_idtype].append(node_id)

    return embeddings_dict, type_to_nodes
class RGCN_LP(nn.Module):
    def __init__(self, in_channels, lattern_channels,hidden_channels, out_channels,num_relations, init_sizes=None):
        super(RGCN_LP, self).__init__()
        self.conv1 = RGCNConv(in_channels, lattern_channels,
                              num_relations=num_relations//2, num_bases=num_relations//2)
        self.conv2 = RGCNConv(lattern_channels, hidden_channels,
                              num_relations=num_relations //2, num_bases=num_relations//2)

        self.conv3 = RGCNConv(hidden_channels, out_channels,
                              num_relations=num_relations//2, num_bases=num_relations//2)
        self.lins = torch.nn.ModuleList()
        for size in init_sizes:
            lin = nn.Linear(size, in_channels)
            self.lins.append(lin)

        self.norm = nn.LayerNorm(out_channels)
        self.fc = nn.Sequential(
            nn.Linear(2 * out_channels, 1)
        )
    def trans_dimensions(self, xs):
        res = []
        for x, lin in zip(xs, self.lins):
            x = x.to(lin.weight.dtype)
            res.append(lin(x))
        return torch.cat(res, dim=0)

    def forward(self, data,init_x):
        x = self.trans_dimensions(init_x)
        edge_index, edge_type = data.edge_index, data.edge_type
        x = F.relu(self.conv1(x, edge_index, edge_type//2)) #DBLP should replace edge_type//2 with edge_type % 3
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type //2))#DBLP should replace edge_type//2 with edge_type % 3
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index, edge_type//2)#DBLP should replace edge_type//2 with edge_type % 3
        x = self.norm(x)
        return x


class ContrastiveLearning(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super(ContrastiveLearning, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU()
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, embeddings_dict, rgcn_emb, neg_src, neg_dst,device=device):
        keys, values = zip(*embeddings_dict.items())
        fairwalk_keys = torch.tensor(keys, dtype=torch.long, device=device)
        fairwalk_values = torch.stack(values, dim=0).to(device)
        fairwalk_values = self.norm(fairwalk_values)
        rgcn_selected = rgcn_emb[fairwalk_keys]
        proj_fairwalk = F.normalize(self.projection(fairwalk_values), dim=-1)
        proj_rgcn = F.normalize(self.projection(rgcn_selected), dim=-1)
        pos_sim = F.cosine_similarity(proj_rgcn, proj_fairwalk, dim=-1)
        neg_rgcn_src = rgcn_emb[neg_src]
        neg_fairwalk_dst = []
        for dst_id in neg_dst.tolist():
            neg_fairwalk_dst.append(embeddings_dict[dst_id])
        neg_fairwalk_dst = torch.stack(neg_fairwalk_dst, dim=0)
        neg_fairwalk_dst = self.norm(neg_fairwalk_dst)
        proj_neg_rgcn_src = F.normalize(self.projection(neg_rgcn_src), dim=-1)
        proj_neg_fairwalk_dst = F.normalize(self.projection(neg_fairwalk_dst), dim=-1)
        neg_sim = F.cosine_similarity(proj_neg_rgcn_src, proj_neg_fairwalk_dst, dim=-1)
        pos_exp = torch.exp(pos_sim)
        neg_exp = torch.exp(neg_sim)
        pos_loss = -torch.log(pos_exp / (pos_exp + neg_exp.sum())).mean()
        return pos_loss

class LinkPrediction(nn.Module):
    def __init__(self, left_right_to_edge_type,node_type_map,embedding_dim=128, hidden_dim=24, batch_size=10000):
        super(LinkPrediction, self).__init__()
        self.left_right_to_edge_type = left_right_to_edge_type
        self.node_type_map=node_type_map
        self.norm = nn.LayerNorm(embedding_dim)
        self.batch_size = batch_size
        edge_type_link=len(left_right_to_edge_type)
        self.discriminative_classifier = nn.ModuleDict({
            str(edge_type): nn.Sequential(
            nn.Linear(embedding_dim* 2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, 1)
        )for edge_type in range(edge_type_link)
        })
    def forward(self, rgcn_emb, edge_label_inde):
        src, dst = edge_label_inde
        src_emb = rgcn_emb[src]
        dst_emb = rgcn_emb[dst]
        edge_emb = torch.cat([src_emb, dst_emb], dim=-1)
        src_types = [self.node_type_map[node.item()] for node in src]
        dst_types = [self.node_type_map[node.item()] for node in dst]
        left_right_key = list(zip(src_types, dst_types))
        edge_type_idx = torch.tensor([self.left_right_to_edge_type.get(key, -1) for key in left_right_key],
                                     device=edge_label_inde.device)
        batch_size=self.batch_size
        num_batches = (edge_type_idx.size(0) + batch_size - 1) // batch_size
        scores_list = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, edge_type_idx.size(0))
            batch_edge_type_idx = edge_type_idx[start_idx:end_idx]
            batch_edge_emb = edge_emb[start_idx:end_idx]
            batch_scores = torch.zeros((batch_edge_type_idx.size(0), 1), device=batch_edge_emb.device)
            unique_edge_types = torch.unique(batch_edge_type_idx)
            for edge_type in unique_edge_types:
                mask = (batch_edge_type_idx == edge_type)
                selected_embs = batch_edge_emb[mask]
                scores = self.discriminative_classifier[str(edge_type.item())](selected_embs)
                batch_scores[mask] = scores

            scores_list.append(batch_scores)

        scores = torch.cat(scores_list, dim=0).squeeze()
        return scores,rgcn_emb



def train_negative_sample_load(train_data,node_type_map,edge_type_to_index,device=device):

    unique_nodes = torch.unique(train_data.edge_index).to(device)
    pos_edges = torch.stack([train_data.edge_index[0], train_data.edge_index[1]], dim=1).to(device)
    pos_edges_set = set(map(tuple, pos_edges.tolist()))
    pos_edges_set.update(tuple(reversed(edge)) for edge in pos_edges_set)
    num_neg_samples = train_data.edge_label_index.size(1)
    neg_edges = []
    neg_edge_types = []
    valid_edge_types = set((etype_src, etype_dst) for etype_src, _, etype_dst in edge_type_to_index.keys())

    while len(neg_edges) < num_neg_samples:
        neg_src = unique_nodes[
            torch.randint(0, unique_nodes.size(0), (num_neg_samples,), device=device)
        ]
        neg_dst = unique_nodes[
            torch.randint(0, unique_nodes.size(0), (num_neg_samples,), device=device)
        ]

        for src, dst in zip(neg_src.tolist(), neg_dst.tolist()):
            if src != dst:
                if (src, dst) not in pos_edges_set and (dst, src) not in pos_edges_set:
                    src_type = node_type_map[src]
                    dst_type = node_type_map[dst]
                    if (src_type, dst_type) in valid_edge_types:
                        edge_type_id = edge_type_to_index.get((src_type, 'relation', dst_type))
                        if edge_type_id is not None:
                            neg_edges.append((src, dst))
                            neg_edge_types.append(edge_type_id)
            if len(neg_edges) >= num_neg_samples:
                break

    neg_src, neg_dst = zip(*neg_edges)
    neg_edge_index = torch.tensor([neg_src, neg_dst], device=device)
    neg_edge_type = torch.tensor(neg_edge_types, device=device)

    return neg_edge_index,neg_edge_type



def load_and_process_training_data(
    best_validate_dir, processed_train_data, init_x,init_sizes, node_type_map,edge_type_to_index,node_type_ranges_homo,homo_node_types,device
):

    embeddings_dict, _ = load_fairwalk_embeddings('./embeddings.emb', processed_train_data.node_type,node_type_ranges_homo,homo_node_types)
    node_counts = [x.shape[0] for x in init_x]
    total_nodes = sum(node_counts)
    left_right_to_edge_type = build_left_right_to_edge_type(edge_type_to_index)

    unique_edge_types = torch.unique(processed_train_data.edge_type)
    num_edge_types = len(unique_edge_types)

    rgcn_model = RGCN_LP(
        in_channels=512,
        lattern_channels=384,
        hidden_channels=256,
        out_channels=128,
        num_relations=num_edge_types,
        init_sizes=init_sizes
    ).to(device)

    link_pred_model = LinkPrediction(
        left_right_to_edge_type,
        node_type_map,
        embedding_dim=128,
        hidden_dim=24
    ).to(device)

    checkpoint = torch.load(best_validate_dir, map_location=device)
    rgcn_model.load_state_dict(checkpoint['rgcn_state_dict'])
    link_pred_model.load_state_dict(checkpoint['link_pred_state_dict'])
    rgcn_model.eval()
    link_pred_model.eval()
    init_x=[x.to(device) for x in init_x]
    with torch.no_grad():
        rgcn_emb = rgcn_model(processed_train_data,init_x)
        neg_edge_index_lp, neg_edge_type = train_negative_sample_load(
            processed_train_data, node_type_map, edge_type_to_index
        )
        combined_edge_label_index = torch.cat([processed_train_data.edge_label_index, neg_edge_index_lp],
                                              dim=1).to(device)
        combined_edge_label = torch.cat(
            [processed_train_data.edge_label, torch.zeros(neg_edge_index_lp.size(1), dtype=torch.int64, device=device)],
            dim=0)

        combined_edge_type = torch.cat([processed_train_data.edge_type, neg_edge_type],
                                       dim=0).to(device)

        shuffled_indices = torch.randperm(combined_edge_label_index.size(1), device=device)
        combined_edge_label_index = combined_edge_label_index[:, shuffled_indices]
        combined_edge_label = combined_edge_label[shuffled_indices]
        combined_edge_type = combined_edge_type[shuffled_indices]
        train_logits,fused_emb = link_pred_model(rgcn_emb, combined_edge_label_index)

    return fused_emb, train_logits, combined_edge_label_index, combined_edge_label, combined_edge_type


