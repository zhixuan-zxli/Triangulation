#ifndef BOVWRITER_H
#define BOVWRITER_H

#include <string>
#include "FiniteDiff/RectDomain.h"

template <class, int> class Tensor;

template <int Dim>
void writeBOV(const RectDomain<Dim> &aGrid,
              const Tensor<Real, Dim> &aData,
              const std::string &folder,
              const std::string &varname,
              int stamp = 0,
              Real instant = 0.0);

// Output of multi-component data. The data is re-arranged so that the component-wise dimension is the major one.
template <int Dim>
void writeBOVByComponent(const RectDomain<Dim> &aGrid,
                         const Tensor<Real, Dim> *aData,
                         int nComp,
                         const std::string &folder,
                         const std::string &varname,
                         int stamp = 0,
                         Real instant = 0.0);

#endif //BOVWRITER_H
