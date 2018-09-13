from utils import run_experiment, label_params

#
# CREATE PARAMS FOR DATASET
#

dataset_params = {
  "name": "epsilon", 
  "inputFile": "/mlodata1/jb/data/epsilon_normalized",
  "numSamples": 400000, 
  "numFeatures": 2000, 
  "is_sparse": False
}

num_features = dataset_params["numFeatures"]
num_samples = dataset_params["numSamples"]

#
# CREATE PARAMS FOR EXPERIMENT
#

common_params = {
  "numEpochs": 3,
  "lrType": "optimal",
  "lr": 1.,
  "cores": 1,
  "lambda_": 1. / num_samples,
  "printPerEpoch": 100,
  "weightingScheme": 'final'
}

params_with_tag = []
tau_no_delay = 10.
tau_delay = tau_no_delay * num_features

params_with_tag.append(label_params('sgd', useMemory=False, takeK=0, takeTop=False, tau=tau_delay, **common_params))
params_with_tag.append(label_params('sgd no delay', useMemory=False, takeK=0, takeTop=False, tau=tau_no_delay, **common_params))

ks = [1, 2, 3]
for i, k in enumerate(ks):
  label = "k={} no delay".format(k)
  params_with_tag.append(label_params(label, useMemory=True, takeK=k, takeTop=False, tau=tau_no_delay, **common_params))
  
  label="k={}".format(k)
  params_with_tag.append(label_params(label, useMemory=True, takeK=k, takeTop=False, tau=tau_delay, **common_params))

# params_with_tag.append(label_params("top k=10", useMemory=True, takeK=10, takeTop=True, tau=tau_delay, **common_params))
params_with_tag.append(label_params("top k=1", useMemory=True, takeK=1, takeTop=True, tau=tau_delay, **common_params))

#
# RUN THE EXPERIMENT
#

run_experiment(dataset_params, params_with_tag)

