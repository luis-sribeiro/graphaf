from torchdrug import core, models, tasks, datasets
from torch import nn, optim
from torchdrug.layers import distribution
import torch
import pickle
import time

download = True

dataset = None

if download == True:
  dataset = datasets.ZINC250k("~/molecule-datasets/", kekulize=True,
                            node_feature="symbol")
  #with open("zinc250k.pkl", "wb") as fout:
  #  pickle.dump(dataset, fout)
else:
  with open("zinc250k.pkl", "rb") as fin:
    dataset = pickle.load(fin)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpus = None

if str(device) == 'cuda':
  print('Available GPU: ', torch.cuda.get_device_name(device))
  gpus = (0,)
else:
  print('There are not GPUs available')
  gpus = None

model = models.RGCN(input_dim=dataset.num_atom_type,
                    num_relation=dataset.num_bond_type,
                    hidden_dims=[256, 256, 256], batch_norm=True)

model = model.to(device)

num_atom_type = dataset.num_atom_type
# add one class for non-edge
num_bond_type = dataset.num_bond_type + 1

node_prior = distribution.IndependentGaussian(torch.zeros(num_atom_type),
                                              torch.ones(num_atom_type))
edge_prior = distribution.IndependentGaussian(torch.zeros(num_bond_type),
                                              torch.ones(num_bond_type))


node_flow = models.GraphAF(model, node_prior, num_layer=12)
edge_flow = models.GraphAF(model, edge_prior, use_edge=True, num_layer=12)

task = tasks.AutoregressiveGeneration(node_flow, edge_flow,
                                      max_node=38, max_edge_unroll=12,
                                      criterion="nll")

optimizer = optim.Adam(task.parameters(), lr = 1e-3)
solver = core.Engine(task, dataset, None, None, optimizer,
                     gpus=gpus, batch_size=128, log_interval=10)                                  

epochs = 10
start_time = time.time()
for epoch in range(epochs):
  solver.train(num_epoch=1)
  solver.save("graphaf_zinc250k_" + str(epoch) + ".pkl")  
  print('Model saved!')
print('Training time: {}'.format(time.time() - start_time))
