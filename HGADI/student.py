import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroLinear, Linear, BatchNorm
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d as BatchNorm, ReLU
from torch.autograd import Variable, Function
import torch.linalg as linalg
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GradReverse(Function):
    @ staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_output):
        return grad_output * -ctx.lambd, None

    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None

class PCABatchNorm(nn.Module):
    def __init__(self, input_dim):

        super(PCABatchNorm, self).__init__()
        self.bn_layer = nn.BatchNorm1d(input_dim)
        self.register_buffer('V_k', None)

    def set_pca_components(self, V_k):
        if self.V_k is not None:
            raise ValueError("PCA components have been set up.")
        self.V_k = V_k

    def forward(self, X):

        X_normalized = self.bn_layer(X)
        if self.V_k is None:
            raise ValueError("PCA component not set up yet.")

        X_transformed = torch.matmul(X_normalized, self.V_k)
        return X_transformed



class Student(nn.Module):

    def __init__(self, input_dim, node_types, transductive_types=None, device=device,pca_bn_dict=None,feature_dim_dict=None):
        super().__init__()
        self.device = device
        self.em_dict = nn.ModuleDict()
        self.mlp_layers = nn.ModuleDict()
        self.bn_layers = nn.ModuleDict()
        self.activation = nn.ReLU()
        self.tt = transductive_types
        self.pca_bn_dict = pca_bn_dict
        self.feature_dim_dict = feature_dim_dict
        self.em_output_dim = {}

        for nt in node_types:
            if transductive_types is not None and nt in transductive_types:
                self.em_dict[nt] = nn.Embedding(transductive_types[nt], 288)
                nn.init.xavier_uniform_(self.em_dict[nt].weight)
                self.em_output_dim[nt] = 288
            else:
                if nt in self.pca_bn_dict:
                    self.em_dict[nt] = self.pca_bn_dict[nt]
                    self.em_output_dim[nt] = input_dim
                else:
                    self.em_dict[nt] = nn.Identity()
                    self.em_output_dim[nt] = self.feature_dim_dict[nt]


            mlp_input_dim = self.em_output_dim[nt]
            self.mlp_layers[nt] = nn.ModuleList([
                Linear(mlp_input_dim, 256),
                Linear(256, 128)
            ])

            self.bn_layers[nt] = nn.ModuleList([
                nn.LayerNorm(256),
                nn.LayerNorm(128)
            ])

    def forward(self, x_dict):
        outputs = {}
        for node_type in x_dict:
            if node_type not in self.em_dict:
                continue
            if isinstance(self.em_dict[node_type], nn.Embedding) or isinstance(self.em_dict[node_type],
                                                                                   PCABatchNorm):
                outputs[node_type] = self.em_dict[node_type](x_dict[node_type].to(self.device).squeeze())
            elif isinstance(self.em_dict[node_type], nn.Identity):
                outputs[node_type] = x_dict[node_type].to(self.device)
            else:
                raise TypeError(f"Unsupported layer type for node type {node_type}.")

            outputs[node_type] = self.mlp_layers[node_type][0](outputs[node_type].to(self.device))
            outputs[node_type] = self.bn_layers[node_type][0](outputs[node_type])
            outputs[node_type] = self.activation(outputs[node_type])
            outputs[node_type] = self.mlp_layers[node_type][1](outputs[node_type])
            outputs[node_type] = self.bn_layers[node_type][1](outputs[node_type])
            outputs[node_type] = self.activation(outputs[node_type])

        return outputs


class AdversarialLinkPredictionModel(nn.Module):
    def __init__(self, stulink_loss, input_dim, device='cpu'):
        super(AdversarialLinkPredictionModel, self).__init__()
        self.stulink_loss = stulink_loss
        self.discriminative_classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.device = device

    def forward(self, trans_train_data, node_embeddings, teacher_emb, teacher_labels, student_labels, node_types_name,
                edge_type_names):

        logits,pos_out,neg_out,link_labels= self.stulink_loss(trans_train_data, node_embeddings, edge_type_names)
        merged_student_embeddings = torch.cat(
            [node_embeddings[node_type].to(self.device) for node_type in node_types_name], dim=0
        )
        embeddings = torch.cat((teacher_emb, merged_student_embeddings), dim=0)
        labels = torch.cat((teacher_labels, student_labels), dim=0)
        indices = torch.randperm(embeddings.size(0))
        shuffled_embeddings = embeddings[indices]
        shuffled_labels = labels[indices]
        reverse_embeddings = GradReverse.apply(shuffled_embeddings, 1.0)
        discriminator_out = self.discriminative_classifier(reverse_embeddings)
        return logits,pos_out,neg_out, link_labels, discriminator_out, shuffled_labels
def stulink_loss(data,x_dict,edge_types):
    logits = []
    labels = []
    loss=0.0
    for edge_type in edge_types:
        edge_type_str = str(edge_type)
        if edge_type not in data.edge_types:
            continue
        edge_label_index = data[edge_type].edge_label_index
        edge_label = data[edge_type].edge_label.long()
        EPS=0.0000001
        if edge_label_index.size(1) == 0:
            continue
        src, trg = edge_label_index
        src_type, _, dst_type = edge_type
        src_emb = x_dict[src_type][src]
        dst_emb = x_dict[dst_type][trg]
        scores_out=(src_emb*dst_emb).sum(dim=-1).view(-1)
        pos_mask = edge_label.bool()
        src_x_pos = src_emb[pos_mask]
        trg_x_pos = dst_emb[pos_mask]
        pos_out = (src_x_pos * trg_x_pos).sum(dim=-1).view(-1)
        neg_mask = ~edge_label.bool()
        src_x_neg = src_emb[neg_mask]
        trg_x_neg = dst_emb[neg_mask]
        neg_out = (src_x_neg * trg_x_neg).sum(dim=-1).view(-1)

        logits.append(scores_out)
        labels.append(edge_label)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    return logits,pos_out,neg_out,labels

def convert_teacher_edges_to_heterodata(
        homo_edge_label_index,
        homo_edge_label,
        homo_edge_type,
        homo_logits,
        homogeneous_to_original_map,
        edge_type_names,
        init_x,
        node_type_ranges_homo,
        homo_node_types
):

    hetero_edge_labels = {edge_type: {'edge_label_index': [], 'edge_label': [], 'teacher_logits': []} for edge_type in
                          edge_type_names}

    num_edges = homo_edge_label_index.size(1)
    for i in range(num_edges):
        src_homo = homo_edge_label_index[0, i].item()
        trg_homo = homo_edge_label_index[1, i].item()
        edge_type_idx = homo_edge_type[i].item()
        edge_type = edge_type_names[edge_type_idx]

        src_ty, _, dst_ty = edge_type

        src_type,src_original_idx = homogeneous_to_original_map[src_homo]
        trg_type,trg_original_idx = homogeneous_to_original_map[trg_homo]
        hetero_edge_labels[edge_type]['edge_label_index'].append([src_original_idx, trg_original_idx])
        hetero_edge_labels[edge_type]['edge_label'].append(homo_edge_label[i].item())
        hetero_edge_labels[edge_type]['teacher_logits'].append(homo_logits[i].item())

    for edge_type in edge_type_names:
        if hetero_edge_labels[edge_type]['edge_label_index']:
            hetero_edge_labels[edge_type]['edge_label_index'] = torch.tensor(
                hetero_edge_labels[edge_type]['edge_label_index'], dtype=torch.long).T
            hetero_edge_labels[edge_type]['edge_label'] = torch.tensor(hetero_edge_labels[edge_type]['edge_label'], dtype=torch.int64)
            hetero_edge_labels[edge_type]['teacher_logits'] = torch.tensor(hetero_edge_labels[edge_type]['teacher_logits'], dtype=torch.float)

    return hetero_edge_labels

def compute_node_embedding_distillation_loss(
        student_embeddings,
        teacher_embeddings,
        teacher_edge_label_index,
        node_types,
        device=device
):

    mse_loss = nn.MSELoss()
    distillation_loss = 0.0
    merged_student_embeddings = torch.cat(
        [student_embeddings[node_type].to(device) for node_type in node_types], dim=0
    )

    student_emb = merged_student_embeddings
    teacher_emb = teacher_embeddings.to(device)
    distillation_loss = mse_loss(student_emb, teacher_emb)

    return distillation_loss

