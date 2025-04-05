import pickle
import os
import torch
import sys
print(sys.executable)
import numpy as np
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import HGBDataset, IMDB, DBLP
from torch_geometric.transforms import RandomLinkSplit
import torch_geometric.transforms as T
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from model import ContrastiveLearning, LinkPrediction,load_fairwalk_embeddings,load_and_process_training_data
from student import Student,compute_node_embedding_distillation_loss,PCABatchNorm,convert_teacher_edges_to_heterodata,AdversarialLinkPredictionModel,stulink_loss
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_path = os.path.abspath(os.path.dirname(os.getcwd()))
import itertools


def save_file(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


transform = T.Compose([
    T.NormalizeFeatures(),
])

def open_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


root = "./data/pakdd2023/"
name = "ACM"

if name == "DBLP":
    dataset = HGBDataset(root + name, name, transform=transform)
    data = dataset.data
    print(data)
    data["paper"]["y2"] = torch.rand(data["paper"].num_nodes) > 0.5
    data['venue'].x = torch.arange(data['venue'].num_nodes).reshape(-1, 1)
    rns = RandomNodeSplit(num_val=0.3, num_test=0, key="y2")
    targets = {"paper"}

elif name == "ACM":
    dataset = HGBDataset(root + name, name, transform=transform)
    data = dataset.data
    print(data)
    data["paper"]["y"] = torch.rand(data["paper"].num_nodes) > 0.5
    data['term'].x = torch.arange(data['term'].num_nodes).reshape(-1, 1)
    rns = RandomNodeSplit(num_val=0.3, num_test=0, key="y")
    targets = {"paper"}

elif name == "IMDB":
    dataset = HGBDataset(root + name, name, transform=transform)
    data = dataset.data
    rns = RandomNodeSplit(num_val=0.3, num_test=0)
    data['keyword'].x = torch.arange(data['keyword'].num_nodes).reshape(-1, 1)
    targets = {"movie"}

splits = rns(data.clone())

from collections import defaultdict

type_to_nodes = defaultdict(list)
for node_type in data.node_types:
    type_to_nodes[node_type] = data[node_type].x.shape[0]

train_nodes = {}
valid_nodes = {}
for nt in data.node_types:
    if nt in targets:
        train_nodes[nt] = set(splits[nt].train_mask.nonzero().flatten().numpy())
        valid_nodes[nt] = set(splits[nt].val_mask.nonzero().flatten().numpy())
data = splits
node_type_ranges = {}
start_idx = 0
for node_type in data.node_types:
    num_nodes = data[node_type].x.size(0)
    node_type_ranges[node_type] = num_nodes
    start_idx += num_nodes

homogeneous_graph = data.to_homogeneous()
num_nodes = homogeneous_graph.num_nodes
num_edges = homogeneous_graph.edge_index.shape[1]
homo_node_types, homo_edge_types = data.metadata()
num_relations = len(homo_edge_types)
node_type_to_nameyingshe = {i: node_type for i, node_type in enumerate(homo_node_types)}

node_type_map = {}
for node_idx, node_type_id in enumerate(homogeneous_graph.node_type.tolist()):
    node_type_map[node_idx] = node_type_to_nameyingshe[node_type_id]

init_sizes = [data[node_type].x.shape[1] for node_type in homo_node_types]
init_x = [data[node_type].x.clone() for node_type in homo_node_types]
homogeneous_to_original = homogeneous_graph.node_type
homogeneous_to_original_map = {}
start_index = 0
for i, node_type in enumerate(homo_node_types):
    num_nodes = node_type_ranges[node_type]
    for j in range(num_nodes):
        homogeneous_index = start_index + j
        homogeneous_to_original_map[homogeneous_index] = (node_type, j)
    start_index += num_nodes
original_to_homogeneous_map = {v: k for k, v in homogeneous_to_original_map.items()}
node_type_ranges_homo = {}
start_index = 0
edge_type_to_index = {etype: idx for idx, etype in enumerate(homo_edge_types)}
edge_type_names = []
for et in data.edge_types:
    edge_type_names.append(et)
node_types_name=[]
for nt in data.node_types:
    node_types_name.append(nt)
transductive_edges = {et: [] for et in data.edge_types}
inductive_edges = {et: [] for et in data.edge_types}
transductive_nodes = {nt: set() for nt in data.node_types}
inductive_nodes = {nt: set() for nt in data.node_types}
for et in data.edge_types:
    st, _, tt = et
    for u, v in data[et].edge_index.numpy().T:
        if st in valid_nodes and u in valid_nodes[st]:
            if tt not in targets :
                inductive_edges[et].append((u, v))
                inductive_nodes[st].add(u)
            continue
        elif tt in valid_nodes and v in valid_nodes[tt]:
            if st not in targets:
                inductive_edges[et].append((u, v))
                inductive_nodes[tt].add(v)
            continue
        else:
            transductive_edges[et].append((u, v))
            transductive_nodes[st].add(u)
            transductive_nodes[tt].add(v)

import torch
from torch_geometric.data import HeteroData

trans_hdata = HeteroData()
ind_hdata = HeteroData()

for k in transductive_nodes:
    transductive_nodes[k] = torch.tensor(list(transductive_nodes[k])).long()
    inductive_nodes[k] = torch.tensor(list(inductive_nodes[k])).long()
    trans_hdata[k].x = data[k].x
    ind_hdata[k].x = data[k].x
    trans_hdata[k].transductive_nodes = transductive_nodes[k]
    ind_hdata[k].inductive_nodes = inductive_nodes[k]

for k in transductive_edges:
    transductive_edges[k] = torch.tensor(list(transductive_edges[k])).long().T
    inductive_edges[k] = torch.tensor(list(inductive_edges[k])).long().T
    trans_hdata[k].edge_index = transductive_edges[k]
    if len(inductive_edges[k])!=0:
        ind_hdata[k].edge_index = inductive_edges[k]

def save_file(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

if os.path.isfile('datasplits/' + 'ind' + name + '_train_data.pickle'):
    train_data = open_file('datasplits/' + 'ind' + name + '_train_data.pickle')
    trans_train_data = open_file('datasplits/' + 'ind' + name + '_train_train_data.pickle')
    trans_valid_data = open_file('datasplits/' + 'ind' + name + '_train_valid_data.pickle')
    valid_data = open_file('datasplits/' + 'ind' + name + '_valid_data.pickle')
    test_data = open_file('datasplits/' + 'ind' + name + '_test_data.pickle')
else:
    rlp = RandomLinkSplit(edge_types=trans_hdata.edge_types,
                          num_val=0.1, num_test=0,add_negative_train_samples=False
    )

    trans_train_data, trans_valid_data, _ = rlp(trans_hdata)
    rlp2 = RandomLinkSplit(
        edge_types=trans_hdata.edge_types,
        num_val=0.0, num_test=0.0,add_negative_train_samples=False
    )

    train_data, _, _ = rlp2(trans_hdata)

    rlp2 = RandomLinkSplit(
        edge_types=ind_hdata.edge_types,
        num_val=0.5, num_test=0.0)

    valid_data, test_data, _ = rlp2(ind_hdata)

    save_file(train_data, 'datasplits/' + 'ind' + name + '_train_data.pickle')
    save_file(trans_train_data, 'datasplits/' + 'ind' + name + '_train_train_data.pickle')
    save_file(trans_valid_data, 'datasplits/' + 'ind' + name + '_train_valid_data.pickle')
    save_file(valid_data, 'datasplits/' + 'ind' + name + '_valid_data.pickle')
    save_file(test_data, 'datasplits/' + 'ind' + name + '_test_data.pickle')


transductive_dict = None
if name == "DBLP":
    transductive_dict = {'venue': data['venue'].num_nodes}
elif name == "IMDB":
    transductive_dict = {'keyword': data['keyword'].num_nodes}
elif name == "ACM":
    transductive_dict = {'term': data['term'].num_nodes}

sys.path.append(os.path.join(os.getcwd(), 'Fairwalk_master'))
import dataprocessing
train_data_path = 'datasplits/ind{name}_train_data.pickle'
train_data_fairwalk = dataprocessing.load_train_data_from_pickle(train_data_path)
graph = dataprocessing.process_train_data_to_graph(train_data_fairwalk, original_to_homogeneous_map)
node_embeddings = dataprocessing.get_edge_embeddings(graph)
def process_train_data(homogeneous_graph, trans_datas,trans_val_datas,original_to_homogeneous_map,init_x):
    edge_type_to_index = {etype: idx for idx, etype in enumerate(homo_edge_types)}
    train_edge_index = []
    train_edge_type = []
    edge_label_list = []

    for edge_type in trans_datas.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = trans_datas[edge_type].edge_index
        print(edge_index)

        for src, dst in edge_index.T:
            src_h = original_to_homogeneous_map[(src_type, int(src))]
            dst_h = original_to_homogeneous_map[(dst_type, int(dst))]
            rel_idx = edge_type_to_index[edge_type]
            train_edge_index.append([src_h, dst_h])
            train_edge_type.append(rel_idx)
    train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).T
    train_edge_type = torch.tensor(train_edge_type, dtype=torch.long)
    mapped_edge_label_index = []
    for edge_type in trans_datas.edge_types:
        src_type, rel, dst_type = edge_type
        edge_label_index = trans_datas[edge_type].edge_index
        num_edges = edge_label_index.size(1)
        edge_label = torch.ones(num_edges, dtype=torch.int64, device=edge_label_index.device)
        for src, dst in edge_label_index.T:
            src_h = original_to_homogeneous_map[(src_type, int(src))]
            dst_h = original_to_homogeneous_map[(dst_type, int(dst))]
            mapped_edge_label_index.append([src_h, dst_h])
        edge_label_list.append(edge_label)
    mapped_edge_label_index = torch.tensor(mapped_edge_label_index, dtype=torch.long).T
    edge_label_list = torch.cat(edge_label_list, dim=0) if edge_label_list else None
    mapped_edge_label_index_val = []
    edge_label_list_val = []
    for edge_type in trans_val_datas.edge_types:
        src_type, rel, dst_type = edge_type
        edge_label_index = trans_val_datas[edge_type].edge_label_index
        edge_label = trans_val_datas[edge_type].edge_label
        for src, dst in edge_label_index.T:
            src_h = original_to_homogeneous_map[(src_type, int(src))]
            dst_h = original_to_homogeneous_map[(dst_type, int(dst))]
            mapped_edge_label_index_val.append([src_h, dst_h])
        edge_label_list_val.append(edge_label)
    mapped_edge_label_index_val = torch.tensor(mapped_edge_label_index_val, dtype=torch.long).T
    edge_label_list_val = torch.cat(edge_label_list_val, dim=0) if edge_label_list_val else None
    x = homogeneous_graph.x
    total_rows_sum = 0
    node_type_list = []
    for i, node_feature_matrix in enumerate(init_x):
        num_rows = node_feature_matrix.shape[0]
        node_type_list.extend([i] * num_rows)
        total_rows_sum += num_rows

    node_type_clone = torch.tensor(node_type_list, dtype=torch.long)

    num_nodes = homogeneous_graph.num_nodes

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for node_type in homo_node_types:
        if node_type in targets:
            original_train_mask = splits[node_type].train_mask
            original_val_mask = splits[node_type].val_mask
            original_test_mask = splits[node_type].test_mask

            for original_idx in torch.nonzero(original_train_mask, as_tuple=False).flatten():
                homogeneous_idx = original_to_homogeneous_map[(node_type, int(original_idx))]
                train_mask[homogeneous_idx] = True

            for original_idx in torch.nonzero(original_val_mask, as_tuple=False).flatten():
                homogeneous_idx = original_to_homogeneous_map[(node_type, int(original_idx))]
                val_mask[homogeneous_idx] = True

            for original_idx in torch.nonzero(original_test_mask, as_tuple=False).flatten():
                homogeneous_idx = original_to_homogeneous_map[(node_type, int(original_idx))]
                test_mask[homogeneous_idx] = True

    processed_train_data = Data(
        edge_index=train_edge_index,
        edge_type=train_edge_type,
        edge_label_index=mapped_edge_label_index,
        edge_label=edge_label_list,
        x=x,
        node_type=node_type_clone.clone(),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    processed_val_data= Data(
        edge_index=train_edge_index,
        edge_type=train_edge_type,
        edge_label_index=mapped_edge_label_index_val,
        edge_label=edge_label_list_val,
        x=x,
        node_type=node_type_clone.clone(),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return processed_train_data, processed_val_data

teacher_train_data, teacher_valid_data = process_train_data(
    homogeneous_graph,
    trans_train_data,
trans_valid_data,
 original_to_homogeneous_map,
    init_x
)



if __name__ == '__main__':
    best_auc = -np.inf
    EPS = 0.00000001
    best_lambda_reg=0.1
    best_model_path = ""
    init_x = [x.to(torch.float32).to(device) for x in init_x]
    # Load the best model and process the training set
    best_model_path = f'best_model/{name}/best_model_lambda_reg_0.50_best_model_epoch_421.pt'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    fused_emb, train_logits,trans_train_edge_label_index,trans_train_edge_label,combined_edge_type= load_and_process_training_data(
        best_model_path, teacher_train_data, init_x,init_sizes,node_type_map,edge_type_to_index, node_type_ranges_homo,homo_node_types,device
    )
    fused_emb = fused_emb.to(device)

    hetero_edge_labels_train = convert_teacher_edges_to_heterodata(
        trans_train_edge_label_index,
        trans_train_edge_label,
        combined_edge_type,
        train_logits,
        homogeneous_to_original_map,
        edge_type_names,
        init_x,
        node_type_ranges_homo,
        homo_node_types
    )

    for edge_type in edge_type_names:
        if hetero_edge_labels_train[edge_type]['edge_label_index'].numel() > 0:
            trans_train_data[edge_type].edge_label_index = hetero_edge_labels_train[edge_type]['edge_label_index'].to(device)
            if trans_train_data[edge_type].edge_label_index.shape[0] != 2:
                raise ValueError(f"edge_label_index for {edge_type} is not in [2, num_edges] format.")
            trans_train_data[edge_type].edge_label = hetero_edge_labels_train[edge_type]['edge_label'].to(device)
        else:
            print("wrong")
            exit(0)
            trans_train_data[edge_type].edge_label_index = torch.empty((2, 0), dtype=torch.long).to(device)
            trans_train_data[edge_type].edge_label = torch.empty((0,), dtype=torch.int64).to(device)

    output_dim = 800
    v_matrices = torch.load(f'{name}_v_matrices.pt')
    pca_bn_dict = {}
    feature_dim_dict = {}
    for node_type in splits.node_types:
        if node_type not in transductive_dict and node_type in v_matrices:
            input_dim = splits[node_type].x.shape[1]
            pca_bn = PCABatchNorm(input_dim)
            pca_bn.set_pca_components(v_matrices[node_type])
            pca_bn_dict[node_type] = pca_bn
        elif node_type not in transductive_dict:
            feature_dim = splits[node_type].x.shape[1]
            feature_dim_dict[node_type] = feature_dim

    student_model = Student(input_dim=output_dim,
                             node_types=node_types_name,
    transductive_types=transductive_dict,
    device= device,
    pca_bn_dict = pca_bn_dict,
    feature_dim_dict=feature_dim_dict
     ).to(device)
    edge_types = edge_type_names
    trans_train_data = trans_train_data.to(device)
    loss_fn = AdversarialLinkPredictionModel(stulink_loss,128).to(device)
    params = [
        {'params': student_model.parameters(), 'lr': 0.001},
        {'params': loss_fn.parameters(), 'lr': 0.0004}
    ]
    optimizer = torch.optim.Adam(
        params,
        lr=0.001
    )
    criterion = nn.MSELoss()
    best_validate_auc = 0.0
    num_epochs = 2500
    alpha=1
    beta=1
    gamma=1
    adversarilloss=0.0
    best_validate_loss = float('inf')
    output_file = './best_studentmodel/'
    os.makedirs(output_file, exist_ok=True)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        student_model.train()
        loss_fn.train()
        optimizer.zero_grad()

        node_embeddings = student_model(trans_train_data.x_dict)
        node_embeddings = {k: v.to(device) for k, v in node_embeddings.items()}


        node_distillation_loss = compute_node_embedding_distillation_loss(node_embeddings, fused_emb.detach(),
                                                                          trans_train_edge_label_index, node_types_name)

        teacher_emb = fused_emb.detach().to(device)
        length_size = teacher_emb.size(0)
        teacher_labels = torch.zeros(length_size,device=device)
        student_labels = torch.ones(length_size,device=device)
        score_logits,pos_out,neg_out,linkst_labels,discriminator_out,st_labels=loss_fn(trans_train_data,node_embeddings,teacher_emb,teacher_labels,student_labels,node_types_name,edge_type_names)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + EPS).mean()
        pos_loss = -torch.log(torch.sigmoid(pos_out) + EPS).mean()
        loss_link = pos_loss + neg_loss
        st_labels = st_labels.unsqueeze(-1)
        adversarial_loss = criterion(discriminator_out, st_labels)
        distillation_loss = 0.0
        mse_logits_loss=0.0
        current_idx = 0
        T = 5
        bce_loss_fn = nn.BCELoss()
        for edge_type in edge_type_names:
            num_edges = trans_train_data[edge_type].edge_label.size(0)
            if num_edges == 0:
                continue

            logits_student = score_logits[current_idx: current_idx + num_edges]
            logits_student = logits_student.unsqueeze(-1)

            logits_teacher = hetero_edge_labels_train[edge_type]['teacher_logits']
            logits_teacher = logits_teacher.detach()
            logits_teacher = logits_teacher.unsqueeze(-1)

            shuffled_indices = torch.randperm(logits_student.size(0), device=device)
            logits_student_shuffled = logits_student[shuffled_indices]
            logits_teacher_shuffled = logits_teacher[shuffled_indices]
            student_prob = torch.sigmoid(logits_student_shuffled / T)
            teacher_prob = torch.sigmoid(logits_teacher_shuffled / T)
            ditillation_los = bce_loss_fn(student_prob, teacher_prob)
            distillation_loss += ditillation_los
            mse_logits_loss+=F.mse_loss(torch.sigmoid(logits_student_shuffled), torch.sigmoid(logits_teacher_shuffled))
            current_idx += num_edges

        total_loss = gamma *loss_link +  node_distillation_loss+ alpha*adversarial_loss + beta*distillation_loss* (T * T)+mse_logits_loss
        total_loss.backward()
        optimizer.step()

        student_model.eval()
        with torch.no_grad():
            node_embeddings = student_model(valid_data.x_dict)
            logits,pos_out_val,neg_out_val,labels_all = stulink_loss(valid_data, node_embeddings,valid_data.edge_types)
            val_loss = -torch.log(1 - torch.sigmoid(neg_out_val) + EPS).mean()-torch.log(torch.sigmoid(pos_out_val) + EPS).mean()
            st_scores_val = logits.cpu().numpy()
            st_labels_val = labels_all.cpu().numpy()
            st_auc_val = roc_auc_score(y_true=st_labels_val, y_score=st_scores_val)
            st_ap_val= average_precision_score(y_true=st_labels_val, y_score=st_scores_val)

            if st_auc_val>best_validate_auc:
                best_validate_auc = st_auc_val
                best_validate_loss = val_loss
                best_validate_dir = os.path.join(output_file, f'best_model_alpha_{alpha:.2f}_best_model_beta_{beta:.2f}_best_model_epoch_{epoch + 1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'student_state_dict': student_model.state_dict(),
                    'adversarial_state_dict': loss_fn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'st_val_loss': best_validate_loss,
                    'st_val_auc': best_validate_auc
                }, best_validate_dir)
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Train Loss: {total_loss.item():.4f}, '
                      f'logits Loss: {distillation_loss.item():.4f}, '
                      f'mide_layer Loss: {node_distillation_loss.item():.4f}, '
                      f'adversarial Loss: {adversarial_loss.item():.4f}, '
                      f'Link Pred Loss: {loss_link.item():.4f}, '
                      f'Val Loss: {val_loss:.4f}, '
                      f'AUC Val: {st_auc_val:.4f}, '
                      f'AP Val: {st_ap_val:.4f}, '
                      f'alpha: {alpha: .2f},'
                      f'beta: {beta: .2f},'
                      )

    print('best_validate_auc',best_validate_auc, 'best_validate_dir',best_validate_dir)

    student_model_test = Student(input_dim=output_dim,
                            node_types=node_types_name,
                            transductive_types=transductive_dict,
                            device=device,
                            pca_bn_dict=pca_bn_dict,
                            feature_dim_dict=feature_dim_dict
                            ).to(device)

    test_data = test_data.to(device)
    checkpoint = torch.load(best_validate_dir, map_location=device)
    student_model_test.load_state_dict(checkpoint['student_state_dict'])
    student_model_test.eval()
    with torch.no_grad():
        node_embeddings = student_model_test(test_data.x_dict)
        test_logits, pos_out_tes, neg_out_tes, labels_test = stulink_loss(test_data, node_embeddings,test_data.edge_types)
        test_loss = -torch.log(1 - torch.sigmoid(neg_out_tes) + EPS).mean() - torch.log(
            torch.sigmoid(pos_out_tes) + EPS).mean()

        st_scores_test = test_logits.cpu().numpy()
        st_labels_test = labels_test.cpu().numpy()
        st_auc_test = roc_auc_score(y_true=st_labels_test, y_score=st_scores_test)
        st_ap_test = average_precision_score(y_true=st_labels_test, y_score=st_scores_test)
    print('Test Loss:',test_loss,'AUC:',st_auc_test,'AP:',st_ap_test)
