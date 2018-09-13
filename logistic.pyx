from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        vector()
        T& operator[](int)
        size_t size()

cdef extern from "logisticSGD.h":
    cdef cppclass Result:
        vector[size_t] ts
        vector[float] losses
        vector[float] timers
        vector[float] memoryNorm
        vector[float] memoryMax
        vector[int] selectedCoordCount

cdef extern from "dataset.h":
    cdef cppclass Dataset:
        Dataset(string, string, size_t, size_t, bool) except +
        void load()
        bool loaded

cdef class PyDataset:
    cdef Dataset* _thisptr

    def __cinit__(self, name, inputFile, numSamples, numFeatures, is_sparse):
        self._thisptr = new Dataset(name.encode(), inputFile.encode(), numSamples, numFeatures, is_sparse)

    def __dealloc__(self):
        del self._thisptr

    def __init__(self, name, inputFile, numSamples, numFeatures, is_sparse):
        pass

    def load(self):
        self._thisptr.load()

    def is_loaded(self):
        return self._thisptr.loaded

cdef extern from "logisticSGD.h":
    cdef cppclass LogisticSGD:
        LogisticSGD(size_t, string, float, float, float, string, bool, float, size_t, bool, size_t, size_t) except +
        Result* fit(Dataset*)
        Result* fit(Dataset*, float)
        float loss(Dataset*)
        float accuracy(Dataset*)

cdef class PygisticSGD:
    cdef LogisticSGD* _thisptr

    def __cinit__(self, numEpochs, lrType, lr, tau, lambda_, weightingScheme, useMemory, memoryBound, takeK, takeTop, cores, printPerEpoch):
        self._thisptr = new LogisticSGD(numEpochs, lrType.encode(), lr, tau, lambda_, weightingScheme.encode(), useMemory, memoryBound, takeK, takeTop, cores, printPerEpoch)

    def __dealloc__(self):
        del self._thisptr

    def __init__(self, numEpochs, lrType, lr, tau, lambda_, weightingScheme, useMemory, memoryBound, takeK, takeTop, cores, printPerEpoch):
        pass

    def fit(self, PyDataset dataset, float until=0):
        assert dataset.is_loaded()
        res = self._thisptr.fit(dataset._thisptr, until)
        return self._result_to_tuple(res)

    def loss(self, PyDataset dataset):
        assert dataset.is_loaded()
        return self._thisptr.loss(dataset._thisptr)

    def accuracy(self, PyDataset dataset):
        assert dataset.is_loaded()
        return self._thisptr.accuracy(dataset._thisptr)

    cdef _result_to_tuple(self, Result* res):
        fields = [res.ts, res.losses, res.timers, res.memoryNorm, res.memoryMax, res.selectedCoordCount]
        results = [[] for _ in range(len(fields))]
        for field, r in zip(fields, results):
            for elem in field:
                r.append(elem)
        return results
