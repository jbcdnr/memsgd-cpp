import os
import argparse
import logistic
import pickle
import multiprocessing as mp

def _run(parameters):
  print(parameters)
  dataset_params, label_params = parameters
  label, params = label_params
  dataset = logistic.PyDataset(**dataset_params)
  dataset.load()
  model = logistic.PygisticSGD(**params)
  print('run model {}'.format(label))
  res = model.fit(dataset)
  print('done model {}'.format(label))
  return label, res 

def label_params(label, **kwargs):
  return label, kwargs

def run_experiment(dataset_params, tag_params_list):
  # make a directory to save
  parser = argparse.ArgumentParser()
  parser.add_argument('--directory', type=str)
  parser.add_argument('--file', type=str)
  args = parser.parse_args()
  directory = args.directory
  file = args.file
  if not os.path.exists(directory):
    os.makedirs(directory)

  # run the models in parallel
  with mp.Pool(len(tag_params_list)) as pool:
    with_dataset = [(dataset_params, p) for p in tag_params_list]
    res = dict(pool.map(_run, with_dataset))
    with open(os.path.join(directory, file), 'wb') as f:
      pickle.dump(res, f)
    print("results saved in {}".format(os.path.join(directory, file)))
