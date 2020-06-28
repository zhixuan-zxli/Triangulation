#ifndef FVOP_H
#define FVOP_H

#include "FiniteDiff/RectDomain.h"
#include "Core/Tensor.h"

template <int Dim>
class LevelOp
{
public:
  LevelOp(const RectDomain<Dim> &aRd) : rd(aRd) { }

  template <class T>
  void computeLaplacian(Real alpha, const Tensor<Real,Dim> &phi, const T &rhs, Tensor<Real,Dim> &LofPhi) const;
  void computeLaplacian(const Tensor<Real,Dim> &phi, Tensor<Real,Dim> &LofPhi) const;
  void computeLaplacian(const Tensor<Real,Dim> &phi, const Tensor<Real,Dim> &rhs, Tensor<Real,Dim> &rsd) const;
  void relaxJacobi(const Tensor<Real,Dim> &phi, const Tensor<Real,Dim> &rhs, Tensor<Real,Dim> &JofPhi, Real w = 1.0) const;
  void relaxGSRB(const Tensor<Real,Dim> &phi, const Tensor<Real,Dim> &rhs, Tensor<Real,Dim> &JofPhi, int pass) const;

  Real computeNorm(const Tensor<Real,Dim> &lhs, int staggered, int nt) const;

  void computeDivergence(const Tensor<Real, Dim> *f, Tensor<Real, Dim> &divf) const;
  void computeGradient(const Tensor<Real, Dim> &f, Tensor<Real, Dim> *df) const;
  void computeCurl(const Tensor<Real, Dim> *u, Tensor<Real, Dim> *curlOfU) const;
  void computeConvection(const Tensor<Real, Dim> *u, Tensor<Real, Dim> *cnvu) const;

  void filterFace2Cell(const Tensor<Real, Dim> *aFaceData, Tensor<Real, Dim> *aCellData) const;

protected:
  template <int d>
  Real conv11(const Tensor<Real, Dim> &F, const Vec<int, Dim> &idx) const;
  template <int d, int m>
  Real conv12(const Tensor<Real, Dim> &F, const Tensor<Real, Dim> &G, const Vec<int, Dim> &idx) const;

  const RectDomain<Dim> rd;
};

#endif //FVOP_H

