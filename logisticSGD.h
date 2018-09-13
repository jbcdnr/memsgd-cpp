#ifndef H_LOGISTIC_SGD
#define H_LOGISTIC_SGD

#define PRINT_PER_EPOCH 100

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <queue>
#include <random>
#include <string>
#include <map>
#include <tuple>
#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "dataset.h"

typedef Eigen::SparseMatrix<float, Eigen::RowMajor> DataMatrix;
typedef Eigen::VectorXf LabelVector;
typedef Eigen::VectorXf WeightVector;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> IndicesMatrix;

typedef std::pair<float, int> P;
typedef std::priority_queue<P, std::vector<P>, std::greater<P>> TopKQueue;

template <class T> 
static inline T sigmoid(T x) { return (tanh(x / 2) + 1) / 2; }

class Result {
  public:
    std::vector<size_t> ts;
    std::vector<float> losses;
    std::vector<float> timers;
    std::vector<float> memoryNorm;
    std::vector<float> memoryMax;
    std::vector<int> selectedCoordCount;

    Result(size_t size, size_t numFeatures) {
      ts = std::vector<size_t>(size + 1);
      losses = std::vector<float>(size + 1);
      timers = std::vector<float>(size + 1);
      memoryNorm = std::vector<float>(size + 1);
      memoryMax = std::vector<float>(size + 1);
      selectedCoordCount = std::vector<int>(numFeatures);
    };
};

const int WEIGHT_FINAL = 0;
const int WEIGHT_UNIFORM = 1;
const int WEIGHT_LINEAR = 2;
const int WEIGHT_SQUARE = 3;

const int LR_CONSTANT = 0;
const int LR_BOTTOU = 1;
const int LR_OPTIMAL = 2;

class LogisticSGD {
  public:
    LogisticSGD(size_t numEpochs, const std::string& lrType, float lr, float tau, float lambda_, const std::string& weightScheme, bool useMemory, float boundMemory, size_t takeK, bool takeTop, size_t cores, size_t printPerEpoch):
      numEpochs(numEpochs), lr(lr), tau(tau), lambda_(lambda_), useMemory(useMemory), boundMemory(boundMemory), takeK(takeK), takeTop(takeTop), cores(cores), printPerEpoch(printPerEpoch) {
        std::random_device rd;
        std::mt19937 g(rd());

        if (weightScheme.compare("final") == 0) {
          this->weightScheme = WEIGHT_FINAL;
        } else if (weightScheme.compare("uniform") == 0) {
          this->weightScheme = WEIGHT_UNIFORM;
        } else if (weightScheme.compare("linear") == 0) {
          this->weightScheme = WEIGHT_LINEAR;
        } else if (weightScheme.compare("square") == 0) {
          this->weightScheme = WEIGHT_SQUARE;
        } else {
          throw std::invalid_argument( "received invalid weighting scheme argument" );
        }

        if (lrType.compare("constant") == 0) {
          this->lrType = LR_CONSTANT;
          this->tau = 0.;
        } else if (lrType.compare("bottou") == 0) {
          this->lrType = LR_BOTTOU;
          this->tau = 1 / (lr * lambda_);
        } else if (lrType.compare("optimal") == 0) {
          this->lrType = LR_OPTIMAL;
        } else {
          throw std::invalid_argument( "received invalid learning rate argument" );
        }
      };
    Result* fit(Dataset* dataset);
    Result* fit(Dataset* dataset, float until);
    float loss(Dataset* dataset);
    float accuracy(Dataset* dataset);
  private:
    // private methods
    inline float learningRate(size_t t);
    void topMemoryCoordinates(TopKQueue* queue, WeightVector* memory);
    IndicesMatrix shuffleIndices(size_t numSamples);

    // optimizer state
    WeightVector w;
    WeightVector wEstimate;

    // randomness generator
    std::mt19937 g;
    std::uniform_int_distribution<> d_uniform;

    // attributes
    size_t numEpochs;
    int lrType;
    float lr;
    float tau;
    float lambda_;
    bool useMemory;
    float boundMemory;
    size_t takeK;
    bool takeTop;
    size_t cores;
    int weightScheme;
    size_t printPerEpoch;
};

#endif

