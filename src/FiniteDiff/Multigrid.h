#ifndef MULTIGRID_H
#define MULTIGRID_H

#include <array>
#include <vector>
#include <cstring>
#include "Core/Tensor.h"
#include "FiniteDiff/RectDomain.h"
#include "FiniteDiff/LevelOp.h"
#include "FiniteDiff/Intergrid.h"
#include "FiniteDiff/GhostFiller.h"

template <int Dim>
class MGLevelOp
{
public:
  MGLevelOp(const RectDomain<Dim> &aDomain, Real aWeightForJacobi, const char *aBcTypes)
      : rd(aDomain), lv(aDomain), gf(aDomain), weightForJacobi(aWeightForJacobi)
  {
    strncpy(bcTypes, aBcTypes, 2*Dim+1);
  }

  virtual ~MGLevelOp() = default;

  virtual Real computeNorm(const Tensor<Real, Dim> &phi) const
  {
    return lv.computeNorm(phi, RectDomain<Dim>::CellCentered, 0);
  }

  virtual void fillGhosts(Tensor<Real, Dim> &phi, const Tensor<Real, Dim> &rhs, const Tensor<Real, Dim-1> *bcData) const
  {
    gf.fillGhosts(phi, bcTypes, bcData);
  }

  virtual void fillGhosts(Tensor<Real, Dim> &phi, const Tensor<Real, Dim> &rhs, int) const
  {
    gf.fillGhosts(phi, bcTypes, 0);
  }

  virtual void applySmoother(const Tensor<Real, Dim> &phi, Tensor<Real, Dim> &smoothed, const Tensor<Real, Dim> &rhs) const
  {
    lv.relaxJacobi(phi, rhs, smoothed, weightForJacobi);
  }

  virtual void computeResidual(const Tensor<Real, Dim> &phi, Tensor<Real, Dim> &rsd, const Tensor<Real, Dim> &rhs) const
  {
    lv.computeLaplacian(phi, rhs, rsd);
  }

  virtual void solveBottom(Tensor<Real, Dim> &phi, const Tensor<Real, Dim> &rhs, int numBottomRelax) const
  {
    Tensor<Real, Dim> tmp(phi.box());
    for(int k = 0; k < numBottomRelax; ++k) {
      fillGhosts(phi, rhs, 0);
      applySmoother(phi, tmp, rhs);
      fillGhosts(tmp, rhs, 0);
      applySmoother(tmp, phi, rhs);
    }
    fillGhosts(phi, rhs, 0);
  }

  const RectDomain<Dim> &getDomain() const { return rd; }

protected:
  const RectDomain<Dim> &rd;
  char bcTypes[2*Dim+1];
  LevelOp<Dim> lv;
  GhostFiller<Dim> gf;
  Real weightForJacobi;
};

//================================================================================

template <int Dim>
class Multigrid
{
public:
  Multigrid(const std::vector<const MGLevelOp<Dim> *> &aOpHier,
            const std::vector<const Intergrid<Dim> *> aTrHier)
      : opHier(aOpHier), trHier(aTrHier)
  {
  }

  void setParam(const std::array<Real, 3> &aRelaxes, int aMaxIter, Real aStallThr, Real aRelTol)
  {
    numRelaxes = aRelaxes;
    maxIter = aMaxIter;
    stallThr = aStallThr;
    relTol = aRelTol;
  }

  Real solve(Tensor<Real,Dim> &phi, const Tensor<Real,Dim> &rhs, const Tensor<Real,Dim-1> *cData) const;

protected:
  template <class T>
  void VCycle(int depth, Tensor<Real,Dim> &phi, const Tensor<Real,Dim> &rhs, const T &cData) const;

  std::vector<const MGLevelOp<Dim> *> opHier;
  std::vector<const Intergrid<Dim> *> trHier;
  std::array<Real, 3> numRelaxes;
  int   maxIter;
  Real  stallThr;
  Real  relTol;
};

#endif // MULTIGRID_H

