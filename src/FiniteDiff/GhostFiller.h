#ifndef GHOSTFILLER_H
#define GHOSTFILLER_H

#include "FiniteDiff/RectDomain.h"
#include "Core/Tensor.h"

template <int Dim>
class GhostFiller
{
public:
  GhostFiller(const RectDomain<Dim> &_rd) : rd(_rd) { }

  // Homogeneous version
  void fillGhosts(Tensor<Real,Dim> &aData, int D, int side, char type, int) const;
  // Nonhomogeneous version
  void fillGhosts(Tensor<Real,Dim> &aData, int D, int side, char type, const Tensor<Real,Dim-1> &cData) const;
  // Batch homogeneous version
  void fillGhosts(Tensor<Real,Dim> &aData, const char *types, int) const;
  // Batch nonhomogeneous version
  void fillGhosts(Tensor<Real,Dim> &aData, const char *types, const Tensor<Real,Dim-1> *cData) const;

  void copyGhosts(const Tensor<Real, Dim> &aSrc, Tensor<Real, Dim> &aDest) const;

protected:
  template <class T>
  void doFillGhosts(Tensor<Real, Dim> &aData, int D, int side, char type, const T &cData) const;

  const RectDomain<Dim> rd;
};

#endif //GHOSTFILLER_H
