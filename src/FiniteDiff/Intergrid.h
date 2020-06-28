#ifndef INTERGRID_H
#define INTERGRID_H

#include "FiniteDiff/RectDomain.h"
#include "Core/Tensor.h"

template <int Dim>
class Intergrid
{
public:
  Intergrid(const RectDomain<Dim> &aCoarseGrid, const RectDomain<Dim> &aFineGrid)
      : coarseGrid(aCoarseGrid), fineGrid(aFineGrid)
  {
    // Multigrid on the staggered grid is not yet supported.
    assert(aCoarseGrid.getStaggered() == RectDomain<Dim>::CellCentered);
    assert(aFineGrid.getStaggered() == RectDomain<Dim>::CellCentered);
  }

  virtual ~Intergrid() = default;

  virtual void applyRestriction(const Tensor<Real,Dim> &fineData, Tensor<Real,Dim> &coarseData, int d = RectDomain<Dim>::CellCentered) const;

  virtual void incrementalProlong(const Tensor<Real,Dim> &coarseData, Tensor<Real,Dim> &fineData) const;

protected:
  const RectDomain<Dim> coarseGrid;
  const RectDomain<Dim> fineGrid;
};

#endif //INTERGRID_H
