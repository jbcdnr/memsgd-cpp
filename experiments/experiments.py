import os
import logistic
import pickle
import numpy as np

directory = "results"
if not os.path.exists(directory):
  os.makedirs(directory)

# create datasets
madelon = logistic.PyDataset("madelon", "/Users/jb/data/madelon.txt", 2000, 500, False)
rcv1 = logistic.PyDataset("rcv1", "/mlodata1/jb/data/rcv1_train.binary", 20242, 47236, True)
rcv1_test = logistic.PyDataset("rcv1_test", "/mlodata1/jb/data/rcv1_test.binary", 677399, 47236, True)
epsilon = logistic.PyDataset("epsilon", "/mlodata1/jb/data/epsilon_normalized", 400000, 2000, True)



# """
# RCV1-test theory
# """
# if not rcv1_test.is_loaded():
#   rcv1_test.load()
#
# num_samples = 677399
# num_features = 47236
# dataset = rcv1_test
# dataset_name = "RCV1"
# file = "rcv1-th.pickle"
# ks = [10, 20, 30]

# models_with_tags = []

# model = logistic.PygisticSGD(numEpochs=3, constantLr=False, lr=2., tau=1., lambda_=1. / num_samples, 
#                              weightingScheme='square', useMemory=False, takeK=0, takeTop=False, cores=1)
# models_with_tags.append(['sgd', model, "C0-"])

# for i, k in enumerate(ks):
#   label = "k={} no delay".format(k)
#   model = logistic.PygisticSGD(numEpochs=3, constantLr=False, lr=2., tau=1., lambda_=1. / num_samples, 
#                                weightingScheme='square', useMemory=True, takeK=k, takeTop=False, cores=1)
#   models_with_tags.append([label, model, "C{}--".format(i + 1)])
  
#   label="k={}".format(k)
#   model = logistic.PygisticSGD(numEpochs=3, constantLr=False, lr=2., tau=47236 / k, lambda_=1. / num_samples, 
#                                weightingScheme='square', useMemory=True, takeK=k, takeTop=False, cores=1)
#   models_with_tags.append([label, model, "C{}-".format(i + 1)])

# label = "top k=10"
# model = logistic.PygisticSGD(numEpochs=3, constantLr=False, lr=2., tau=47236 / 10, lambda_=1. / num_samples, 
#                              weightingScheme='square', useMemory=True, takeK=10, takeTop=True, cores=1)
# models_with_tags.append([label, model, "C4-"])

# results = {}
# dataset = rcv1_test

# for label, model, color in models_with_tags:
#   print('run model {} on {}'.format(label, dataset_name))
#   results[label] = model.fit(dataset)
#   with open(os.path.join(directory, file), 'wb') as f:
#     pickle.dump(results, f)
#   print('done')



# """
# epsilon theory
# """
# if not epsilon.is_loaded():
#   epsilon.load()
#


# num_samples = 400000
# num_features = 2000
# dataset = rcv1_test
# dataset_name = "epsilon"
# file = "epsilon-th.pickle"
# ks = [1, 2, 3]

# models_with_tags = []

# model = logistic.PygisticSGD(numEpochs=3, constantLr=False, lr=2., tau=1., lambda_=1. / num_samples, 
#                              weightingScheme='square', useMemory=False, takeK=0, takeTop=False, cores=1)
# models_with_tags.append(['sgd', model, "C0-"])

# for i, k in enumerate(ks):
#   label = "k={} no delay".format(k)
#   model = logistic.PygisticSGD(numEpochs=3, constantLr=False, lr=2., tau=1., lambda_=1. / num_samples, 
#                                weightingScheme='square', useMemory=True, takeK=k, takeTop=False, cores=1)
#   models_with_tags.append([label, model, "C{}--".format(i + 1)])
  
#   label="k={}".format(k)
#   model = logistic.PygisticSGD(numEpochs=3, constantLr=False, lr=2., tau=47236 / k, lambda_=1. / num_samples, 
#                                weightingScheme='square', useMemory=True, takeK=k, takeTop=False, cores=1)
#   models_with_tags.append([label, model, "C{}-".format(i + 1)])

# label = "top k=1"
# model = logistic.PygisticSGD(numEpochs=3, constantLr=False, lr=2., tau=47236 / 1, lambda_=1. / num_samples, 
#                              weightingScheme='square', useMemory=True, takeK=1, takeTop=True, cores=1)
# models_with_tags.append([label, model, "C4-"])

# results = {}
# dataset = rcv1_test

# for label, model, color in models_with_tags:
#   print('run model {} on {}'.format(label, dataset_name))
#   results[label] = model.fit(dataset)
#   with open(os.path.join(directory, file), 'wb') as f:
#     pickle.dump(results, f)
#   print('done')


# """
# RCV1 test multi core
# """
# if not rcv1_test.is_loaded():
#   rcv1_test.load()

# num_samples = 677399
# num_features = 47236
# dataset = rcv1_test
# dataset_name = "RCV1"
# file = "rcv1-par.pickle"
# cores = [1,2,5,10,15,20,25]
# repeat = 3
# baseline = 0.101

# models = [
#     ("rand100", lambda core: logistic.PygisticSGD(numEpochs=5, constantLr=False, lr=2., tau=num_features, 
#                                                 lambda_=1. / num_samples, weightingScheme='final', useMemory=True, 
#                                                 takeK=10, takeTop=False, cores=core)),
#     ("top100", lambda core: logistic.PygisticSGD(numEpochs=5, constantLr=False, lr=2., tau=num_features, 
#                                                 lambda_=1. / num_samples, weightingScheme='final', useMemory=True, 
#                                                 takeK=10, takeTop=True, cores=core)),
#     ("sgd", lambda core: logistic.PygisticSGD(numEpochs=5, constantLr=False, lr=2., tau=1, 
#                                                 lambda_=1. / num_samples, weightingScheme='final', useMemory=False, 
#                                                 takeK=0, takeTop=False, cores=core)),
# ]


# all_ts = np.zeros((len(models), len(cores), repeat))
# all_timers = np.zeros((len(models), len(cores), repeat))

# for rep in range(repeat):
#   for j, core in enumerate(cores):
#     for i, (name, model) in enumerate(models):
#       print("run model {} with cores {} repeat {} on {}".format(name, core, rep, dataset_name))
#       m = model(core)
#       ts, losses, timers = m.fit(dataset, baseline)

#       reached_baselines = np.argwhere(np.array(losses) < baseline)
#       if len(reached_baselines) > 0:
#         index = reached_baselines[0,0]
#         all_ts[i, j, rep] = ts[index]
#         all_timers[i, j, rep] = timers[index]
      
#       results = {
#         'models': [m[0] for m in models],
#         'cores': cores,
#         'repeat': repeat,
#         'ts': all_ts,
#         'timers': all_timers, 
#       }
#       with open(os.path.join(directory, file), 'wb') as f:
#         pickle.dump(results, f)
#       print('done')



"""
epsilon multi core
"""
if not epsilon.is_loaded():
  epsilon.load()

num_samples = 400000
num_features = 2000
dataset = epsilon
dataset_name = "epsilon"
file = "epsilon-par.pickle"
cores = [1,2,5,10,15,20,25]
repeat = 3
baseline = 0.305

models = [
    ("rand1", lambda core: logistic.PygisticSGD(numEpochs=5, constantLr=True, lr=.05, tau=num_features, 
                                                lambda_=1. / num_samples, weightingScheme='final', useMemory=True, 
                                                takeK=1, takeTop=False, cores=core)),
    ("top1", lambda core: logistic.PygisticSGD(numEpochs=5, constantLr=True, lr=.05, tau=num_features, 
                                                lambda_=1. / num_samples, weightingScheme='final', useMemory=True, 
                                                takeK=1, takeTop=True, cores=core)),
    ("sgd", lambda core: logistic.PygisticSGD(numEpochs=5, constantLr=True, lr=.05, tau=1, 
                                                lambda_=1. / num_samples, weightingScheme='final', useMemory=False, 
                                                takeK=0, takeTop=False, cores=core)),
]


all_ts = np.zeros((len(models), len(cores), repeat))
all_timers = np.zeros((len(models), len(cores), repeat))

for rep in range(repeat):
  for j, core in enumerate(cores):
    for i, (name, model) in enumerate(models):
      print("run model {} with cores {} repeat {} on {}".format(name, core, rep, dataset_name))
      m = model(core)
      ts, losses, timers = m.fit(dataset, baseline)

      reached_baselines = np.argwhere(np.array(losses) < baseline)
      if len(reached_baselines) > 0:
        index = reached_baselines[0,0]
        all_ts[i, j, rep] = ts[index]
        all_timers[i, j, rep] = timers[index]
      
      results = {
        'models': [m[0] for m in models],
        'cores': cores,
        'repeat': repeat,
        'ts': all_ts,
        'timers': all_timers, 
      }
      with open(os.path.join(directory, file), 'wb') as f:
        pickle.dump(results, f)
      print('done')



