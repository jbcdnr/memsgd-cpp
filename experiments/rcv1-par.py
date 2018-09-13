"""
RCV1 test multi core
"""

from utils import run_experiment, label_params
import logistic
import numpy as np
import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str)
parser.add_argument('--file', type=str)
args = parser.parse_args()
directory = args.directory
file = args.file
if not os.path.exists(directory):
  os.makedirs(directory)

#
# CREATE PARAMS FOR DATASET
#

dataset_params = {
  "name": "rcv1_test", 
  "inputFile": "/mlodata1/jb/data/rcv1_test.binary",
  "numSamples": 677399, 
  "numFeatures": 47236, 
  "is_sparse": True
}

num_features = dataset_params["numFeatures"]
num_samples = dataset_params["numSamples"]

dataset = logistic.PyDataset(**dataset_params)


#
# CREATE PARAMS FOR EXPERIMENT
#

common_params = {
  "numEpochs": 20,
  "lrType": "optimal",
  "lr": 1.,
  "tau": 10 * num_features,
  "lambda_": 1. / num_samples,
  "printPerEpoch": 20,
  "weightingScheme": 'final',
}

params_with_tag = []

params_with_tag.append(label_params('sgd', useMemory=False, takeK=0, takeTop=False, **common_params))
params_with_tag.append(label_params('rand100', useMemory=True, takeK=100, takeTop=False, **common_params))
params_with_tag.append(label_params('top1', useMemory=True, takeK=1, takeTop=True, **common_params))

## RUN

if not dataset.is_loaded():
  dataset.load()

cores = [1, 5, 10, 15, 20, 25]
repeat = 3
baseline = 0.0854

all_ts = np.zeros((len(params_with_tag), len(cores), repeat))
all_timers = np.zeros((len(params_with_tag), len(cores), repeat))

for rep in range(repeat):
  for j, core in enumerate(cores):
    for i, (name, params) in enumerate(params_with_tag):
      print("run model {} with cores {} repeat {}".format(name, core, rep))
      m = logistic.PygisticSGD(cores=core, **params)
      ts, losses, timers, _, _, _ = m.fit(dataset, baseline)

      reached_baselines = np.argwhere(np.array(losses) < baseline)
      if len(reached_baselines) > 0:
        index = reached_baselines[0,0]
        all_ts[i, j, rep] = ts[index]
        all_timers[i, j, rep] = timers[index]
      
      results = {
        'models': [m[0] for m in params_with_tag],
        'cores': cores,
        'repeat': repeat,
        'ts': all_ts,
        'timers': all_timers, 
      }

      with open(os.path.join(directory, file), 'wb') as f:
        pickle.dump(results, f)
      print("results saved in {}".format(os.path.join(directory, file)))
