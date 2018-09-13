#ifndef H_DATASET
#define H_DATASET

#include <string>
#include <memory>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

typedef Eigen::SparseMatrix<float, Eigen::RowMajor> DataMatrix;
typedef Eigen::VectorXf LabelVector;

class Dataset {
  public:
    std::string name, inputFile;
    size_t numSamples, numFeatures;
    bool is_sparse;

    std::unique_ptr<DataMatrix> X;
    std::unique_ptr<LabelVector> y;
    bool loaded;

    Dataset(std::string name, std::string inputFile, size_t numSamples, size_t numFeatures, bool is_sparse):
      name(name), inputFile(inputFile), numSamples(numSamples), numFeatures(numFeatures), is_sparse(is_sparse), loaded(false) {};
    void load();
};

#endif
