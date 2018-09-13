CXX = g++-7
INCLUDE = include/
FLAGS = -std=c++17 -fopenmp -lstdc++ -O3 -Wall
FILES = main.cpp logisticSGD.cpp dataset.cpp

all:
	$(CXX) $(FLAGS) -isystem $(INCLUDE) $(FILES) -o bin/memsgd

python:
	python3 setup.py build_ext
	pip3 install . --user --upgrade

clean:
	rm -rf build/ logistic.cpp bin/*
