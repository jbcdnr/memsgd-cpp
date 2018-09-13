#include "logisticSGD.h"
#include "dataset.h"

#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>
#include <queue>
#include <vector>
#include <memory>

#define SPARSE_MEMORY false

Result* LogisticSGD::fit(Dataset* dataset) {
  return fit(dataset, 0.);
}

Result* LogisticSGD::fit(Dataset* dataset, float until) {
  size_t numFeatures = dataset->X->cols();
  size_t numSamples = dataset->X->rows();

  std::cout << "initialize weights" << std::endl;
  w = Eigen::MatrixXf::Random(numFeatures, 1) / 100;
  wEstimate = 1. * w;
  Eigen::SparseVector<float> dw(numFeatures);
  WeightVector memory = Eigen::MatrixXf::Zero(numFeatures, 1);

  std::uniform_int_distribution<> d_uniform(0, numFeatures - 1);
  
  IndicesMatrix indices = shuffleIndices(numSamples);
  size_t sampleIndex = 0;

  const auto startTime = std::chrono::system_clock::now();
  const size_t print_every = printPerEpoch > 0 ? numSamples / printPerEpoch : 0;
  float lr = 0.;
  size_t t = 0;
  float rho = 0.;
  TopKQueue queue;

  Eigen::VectorXi lastUpdated = Eigen::VectorXi::Zero(numFeatures);

  // results storage
  Result* results = new Result(numEpochs * printPerEpoch, numFeatures);
  for (size_t i=0; i < numEpochs * printPerEpoch + 1; i++) {
    results->ts[i] = i * print_every;
  }

  // prefill scale
  Eigen::VectorXf scales(numEpochs * numSamples + 1);
  scales[0] = 1.;
  for (size_t i=1; i < numEpochs * numSamples + 1; i++) {
    scales[i] = (1 - learningRate(i - 1) * lambda_) * scales[i - 1];
  }

  // prefill learning rate and cum learning rates
  Eigen::VectorXf lrs(numEpochs * numSamples);
  Eigen::VectorXf cumLr(numEpochs * numSamples);
  lrs[0] = learningRate(0);
  cumLr[0] = learningRate(0);
  for (size_t i=1; i < numEpochs * numSamples; i++) {
    lrs[i] = learningRate(i);
    cumLr[i] = cumLr[i-1] + lrs[i];
  }

  omp_set_dynamic(0);
  omp_set_num_threads(cores);

  bool stopFlag = false;

  std::cout << "starts training 2" << std::endl;
#pragma omp parallel for firstprivate(memory, dw, t, lr, sampleIndex, queue, rho) shared(w, wEstimate, stopFlag) schedule(dynamic) collapse(2)
  for (size_t epoch = 0; epoch < numEpochs; epoch++)
  {
    for (size_t iteration = 0; iteration < numSamples; iteration++)
    {
      if (stopFlag) {
        continue;
      }   

      if (iteration == 0)
      {
        std::cout << "=== epoch " << epoch << " ===" << std::endl;
      }

      if ((printPerEpoch > 0 && iteration % print_every == 0) || iteration == numSamples - 1)
      {
        if(weightScheme == WEIGHT_FINAL) {
          if (!useMemory) {
            wEstimate = scales[t] * w;
          } else {
            wEstimate = 1. * w;
          }
        }

        std::chrono::duration<double> diff = std::chrono::system_clock::now() - startTime;
        auto l = loss(dataset);
        auto tt = epoch * printPerEpoch + iteration / print_every;
        results->timers[tt] = diff.count();
        results->losses[tt] = l;
        results->memoryNorm[tt] = memory.norm();
        results->memoryMax[tt] = memory.cwiseAbs().maxCoeff();
        std::cout << "epoch " << epoch << " iter " << iteration << " loss " << l << " timer " << diff.count() << "s" << std::endl;

        if (l < until) {
          stopFlag = true;
        }
      }

      sampleIndex = indices(epoch, iteration);
      t = epoch * numSamples + iteration;
      lr = lrs[t];
      
      auto xi = dataset->X->innerVector(sampleIndex);
      auto yi = (*dataset->y)(sampleIndex);

      if (!useMemory)
      {
        dw = -yi * xi * sigmoid(-yi * xi.dot(scales[t] * w));
        w -= lr * dw / scales[t+1];
      }
      else // with memory
      {
        dw = -yi * xi * sigmoid(-yi * xi.dot(w));
        memory += lr * dw;
        if (!SPARSE_MEMORY) {
          memory += w * lambda_ * lr;
        }

        if (boundMemory > 0) {
          memory = memory.cwiseMin(boundMemory).cwiseMax(-boundMemory);
        }

        if (takeTop)  // top k sparsifier
        {
          topMemoryCoordinates(&queue, &memory);
        }

        size_t coordinate;
        for (size_t k = 0; k < takeK; k++)
        {
          if (takeTop) {
            coordinate = queue.top().second;
            queue.pop();
          } else {
            coordinate = d_uniform(g);
          }

          if (SPARSE_MEMORY) {
            auto coordLastUpdated = lastUpdated[coordinate];
            float regularizer = std::max(0.f, 1 - lambda_ * (cumLr[t] - cumLr[coordLastUpdated]) / cores);
            w[coordinate] *= regularizer;
            lastUpdated[coordinate] = t;
          }
          w[coordinate] -= memory(coordinate);
          memory[coordinate] = 0.;
          results->selectedCoordCount[coordinate] += 1;
        }
      }

      // estimate update
      auto scale = scales[t];
      if (useMemory) {
        scale = 1.;
      }
      switch(weightScheme) {
        case WEIGHT_FINAL:
          rho = 1.;
          break;
        case WEIGHT_UNIFORM:
          rho = 1 / (t + 1);
          break;
        case WEIGHT_LINEAR:
          rho = 2 * (t + tau) / ((1 + t) * (t + 2 * tau));
          break;
        case WEIGHT_SQUARE:
          rho = 6 * ((t + tau) * (t + tau)) / ((1 + t) * (6 * (tau * tau) + t + 6 * tau * t + 2 * (t * t)));
          break;
      }
    }
  }

  if(weightScheme == WEIGHT_FINAL) {
    wEstimate = scales[t] * w;
  }

  return results;
}

float LogisticSGD::loss(Dataset* dataset) {
  float loss = (1 + (-dataset->y->array() * ((*dataset->X) * wEstimate).array()).exp()).log().sum() / dataset->X->rows();
  loss += lambda_ * wEstimate.squaredNorm() / 2;
  return loss;
}

float LogisticSGD::accuracy(Dataset* dataset) {
  float correct = ((*dataset->X) * wEstimate).array().sign().cwiseEqual(dataset->y->array()).cast<float>().sum();
  return correct / dataset->X->rows();
}

inline float LogisticSGD::learningRate(size_t t) {
  if (lrType == LR_CONSTANT) {
    return lr;
  } else if (lrType == LR_BOTTOU) {
    return lr / (1 + lr * lambda_ * t);
  } else { // optimal
    return lr / (lambda_ * (t + tau));
  }
}

void LogisticSGD::topMemoryCoordinates(TopKQueue* queue, WeightVector* memory) {
  for (int d = 0; d < w.rows(); d++)
  {
    if (queue->size() < takeK)
    {
      queue->push(std::make_pair(abs((*memory)(d)), d));
    }
    else if (abs((*memory)(d)) > queue->top().first)
    {
      queue->pop();
      queue->push(std::make_pair(abs((*memory)(d)), d));
    }
  }
}

IndicesMatrix LogisticSGD::shuffleIndices(size_t numSamples) {
  std::cout << "shuffling batches" << std::endl;
  IndicesMatrix indices(numEpochs, numSamples);
  
  #pragma omp parallel for collapse(2)
  for (size_t e = 0; e < numEpochs; e++)
  {
    for (size_t n = 0; n < numSamples; n++)
    {
      indices(e, n) = n;
    }
  }
  for (size_t e = 0; e < numEpochs; e++)
  {
    std::shuffle(indices.row(e).data(), indices.row(e).data() + numSamples, g);
  }
  return indices;
}

