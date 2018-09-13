#include "dataset.h" 

#include <iostream>

void Dataset::load() {
  if (loaded) {
    std::cout << "dataset already loaded" << std::endl;
    return;
  }

  std::cout << "loading dataset from " << inputFile << std::endl;
  std::ifstream input(inputFile);

  typedef Eigen::Triplet<float> T;
  std::vector<T> tripletList;
  std::vector<float> labelsList;

  int row = -1;
  size_t num_features(0);
  for (std::string line; std::getline(input, line);)
  {
    size_t pos = line.find_first_of(" ");
    ++row;
    labelsList.push_back((float) std::stoi(line.substr(0, pos)));

    size_t next_pos = 0;
    size_t column;
    float value;
    while (pos != std::string::npos)
    {
      next_pos = line.find_first_of(':', pos);
      column = std::atoi(line.substr(pos, next_pos - pos).c_str());
      num_features = std::max(num_features, column);

      pos = next_pos;
      next_pos = line.find_first_of(' ', pos);
      value = std::atof(line.substr(pos + 1, next_pos - pos).c_str());

      tripletList.push_back(T(row, column - 1, value));
      pos = next_pos;
    }
  }
  size_t num_samples = labelsList.size();
  X = std::make_unique<DataMatrix>(num_samples, num_features);
  X->setFromTriplets(tripletList.begin(), tripletList.end());

  y = std::make_unique<Eigen::VectorXf>(num_samples);
  for(size_t i=0; i < num_samples; i++){
    (*y)(i) = labelsList[i];
  }

  std::cout << "loaded X(" << X->rows() << "," << X->cols() << ") "
            << "y(" << y->rows() << ")" << std::endl;

  loaded = true;
}