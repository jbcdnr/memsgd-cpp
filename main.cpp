#include "dataset.h"
#include "logisticSGD.h"

int main()
{
  Dataset madelon = {"madelon", "/Users/jb/data/madelon.txt", 2000, 500, false};
  Dataset rcv1 = {"rcv1", "/mlodata1/jb/data/rcv1_train.binary", 20242, 47236, true};
  Dataset rcv1_test = {"rcv1_test", "/mlodata1/jb/data/rcv1_test.binary", 677399, 47236, true};
  Dataset epsilon = {"epsilon", "/mlodata1/jb/data/epsilon_normalized", 400000, 2000, false};
  Dataset* dataset = &rcv1_test;

  // initialisation
  LogisticSGD classifier = LogisticSGD(
    10, // numEpochs
    "optimal", // lrType
    10., // lr
    dataset->numFeatures, // tau
    // 1., // tau
    1. / dataset->numSamples, // lambda
    "final", // weighting scheme
    true, // useMemory
    0, // memoryBound
    10, // takeK
    false, // takeTop
    2, // cores
    10 // PRINT_PER_EPOCH
  );

  dataset->load();

  auto r = classifier.fit(dataset);

  std::cout << "ts ";
  for (auto i = r->ts.begin(); i != r->ts.end(); ++i) 
    std::cout << *i << " ";
  std::cout << std::endl;

  std::cout << "losses ";
  for (auto i = r->losses.begin(); i != r->losses.end(); ++i) 
    std::cout << *i << " ";
  std::cout << std::endl;

  std::cout << "timers ";
  for (auto i = r->timers.begin(); i != r->timers.end(); ++i) 
    std::cout << *i << " ";
  std::cout << std::endl;

  delete r;
}
