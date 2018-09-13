import logistic
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_svmlight_file
import time
import numpy as np

repeat = 3

# #RVV1
# rcv1_test = logistic.PyDataset("rcv1_test", "/mlodata1/jb/data/rcv1_test.binary", 677399, 47236, True)
# svm_path = "/mlodata1/jb/data/rcv1_test.binary"
# dataset = rcv1_test
# num_samples = 677399
# print("RUN ON RCV1")

#epsilon
epsilon = logistic.PyDataset("epsilon", "/mlodata1/jb/data/epsilon_normalized", 400000, 2000, True)
svm_path = "/mlodata1/jb/data/epsilon_normalized"
dataset = epsilon
num_samples = 400000
print("RUN ON epsilon")

dataset.load()

res = np.zeros(repeat)
for i in range(repeat):
    model = logistic.PygisticSGD(numEpochs=1, lrType="bottou", lr=.1, tau=1., lambda_=1. / num_samples, 
                         weightingScheme='final', useMemory=False, takeK=0, takeTop=False, cores=1, printPerEpoch=1)
    print("start {}".format(i))
    start = time.time()
    model.fit(dataset)
    elapsed = time.time() - start
    res[i] = elapsed
    print("done {} took {}s".format(i, elapsed))
    
print("*" * 10)
print("rcv1 | c++ | sgd | {0:.2f} \pm {1:.2f}s".format(res.mean(), res.std()))
print("*" * 10)


res = np.zeros(repeat)
for i in range(repeat):
    model = logistic.PygisticSGD(numEpochs=1, lrType="bottou", lr=.1, tau=1., lambda_=1. / num_samples, 
                         weightingScheme='square', useMemory=False, takeK=0, takeTop=False, cores=1, printPerEpoch=1)
    print("start {}".format(i))
    start = time.time()
    model.fit(dataset)
    elapsed = time.time() - start
    res[i] = elapsed
    print("done {} took {}s".format(i, elapsed))
    
print("*" * 10)
print("rcv1 | c++ | averaged sgd | {0:.2f} \pm {1:.2f}s".format(res.mean(), res.std()))
print("*" * 10)


num_samples = 677399
res = np.zeros(repeat)
for i in range(repeat):
    model = logistic.PygisticSGD(numEpochs=1, lrType="bottou", lr=.1, tau=1., lambda_=1. / num_samples, 
                         weightingScheme='final', useMemory=True, takeK=1, takeTop=False, cores=1, printPerEpoch=1)
    print("start {}".format(i))
    start = time.time()
    model.fit(dataset)
    elapsed = time.time() - start
    res[i] = elapsed
    print("done {} took {}s".format(i, elapsed))
    
print("*" * 10)
print("rcv1 | c++ | mem sgd | {0:.2f} \pm {1:.2f}s".format(res.mean(), res.std()))
print("*" * 10)


res = np.zeros(repeat)
for i in range(repeat):
    model = logistic.PygisticSGD(numEpochs=1, lrType="bottou", lr=.1, tau=1., lambda_=1. / num_samples, 
                         weightingScheme='square', useMemory=True, takeK=1, takeTop=False, cores=1, printPerEpoch=1)
    print("start {}".format(i))
    start = time.time()
    model.fit(dataset)
    elapsed = time.time() - start
    res[i] = elapsed
    print("done {} took {}s".format(i, elapsed))
    
print("*" * 10)
print("rcv1 | c++ | averaged mem sgd | {0:.2f} \pm {1:.2f}s".format(res.mean(), res.std()))
print("*" * 10)


# print("load svm")
# dataset_svm = load_svmlight_file(svm_path)
# X, y = dataset_svm

# res = np.zeros(repeat)
# for i in range(repeat):
#     clf = SGDClassifier(max_iter=1, loss='log', penalty='l2', alpha=1. / X.shape[0], fit_intercept=False, verbose=True)
#     print("start {}".format(i))
#     start = time.time()
#     clf.fit(X,y)
#     elapsed = time.time() - start
#     res[i] = elapsed
#     print("done {} took {}s".format(i, elapsed))
    
# print("*" * 10)
# print("rcv1 | sklearn | sgd | {0:.2f} \pm {1:.2f}s".format(res.mean(), res.std()))
# print("*" * 10)


# res = np.zeros(repeat)
# for i in range(repeat):
#     clf = SGDClassifier(max_iter=1, loss='log', penalty='l2', alpha=1. / X.shape[0], fit_intercept=False, verbose=True, average=True)
#     print("start {}".format(i))
#     start = time.time()
#     clf.fit(X,y)
#     elapsed = time.time() - start
#     res[i] = elapsed
#     print("done {} took {}s".format(i, elapsed))
    
# print("*" * 10)
# print("rcv1 | sklearn | average sgd | {0:.2f} \pm {1:.2f}s".format(res.mean(), res.std()))
# print("*" * 10)
