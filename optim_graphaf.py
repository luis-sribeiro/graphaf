import torch
from torchdrug import core, models, tasks, datasets
from torch import nn, optim
from torchdrug.layers import distribution
from collections import defaultdict
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
                                      task="qed",
                                      criterion={"ppo": 0.25, "nll": 1.0},
                                      reward_temperature=10, baseline_momentum=0.9,
                                      agent_update_interval=5, gamma=0.9)


optimizer = optim.Adam(task.parameters(), lr=1e-5)
solver = core.Engine(task, dataset, None, None, optimizer,
                     gpus=gpus, batch_size=64, log_interval=10)


solver.load("graphaf_zinc250k_9.pkl", load_optimizer=False)
epochs = 10
start_time = time.time()
for epoch in range(epochs):
  solver.train(num_epoch=1)
  solver.save("graphaf_zinc250k_finetune_" + str(epoch) + ".pkl")  
  print('Model saved!')                 
print('Training time: {}'.format(time.time() - start_time))
results = task.generate(num_sample=32, max_resample=5)
print(results.to_smiles())
