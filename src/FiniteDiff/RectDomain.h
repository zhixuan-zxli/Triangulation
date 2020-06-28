#ifndef RECTGRID_H
#define RECTGRID_H

#include <array>
#include "Core/Box.h"

template <int Dim>
class RectDomain : public Box<Dim>
{
public:
  using rVec = Vec<Real,Dim>;
  using iVec = Vec<int,Dim>;
  using BaseClass = Box<Dim>;

  enum StaggerType { CellCentered = -1 };

  RectDomain() = default;

  RectDomain(const BaseClass &aBox, const rVec& aDx, int aStaggered, int aNumGhost)
      : BaseClass(aBox), dx(aDx), staggered(aStaggered), nGhost(aNumGhost)
  {
  }

  // accessor
public:
  using BaseClass::lo;
  using BaseClass::hi;
  using BaseClass::size;
  using BaseClass::volume;
  //
  const rVec &spacing() const { return dx; }
  const rVec getDelta() const {
    rVec delta = 0.5;
    if(staggered != CellCentered)
      delta[staggered] = 0;
    return delta;
  }
  int getStaggered() const { return staggered; }
  int getNumGhost() const { return nGhost; }
  //
  BaseClass getGhostedBox() const { return BaseClass::inflate(nGhost); }

  // deformation
public:
  RectDomain refine() const {
    iVec newlo = lo() * 2;
    iVec newhi = hi() * 2 + 1;
    if(staggered != CellCentered)
      --newhi[staggered];
    return RectDomain(BaseClass(newlo, newhi), dx/2, staggered, nGhost);
  }
  RectDomain coarsen() const {
    iVec newlo = lo() / 2;
    iVec newhi = hi() / 2;
//    if(staggered != CellCentered)
//      ++newlo[staggered];
    return RectDomain(BaseClass(newlo, newhi), dx*2, staggered, nGhost);
  }
  RectDomain stagger(int type) const {
    if(type == staggered)
      return *this;
    if(staggered == CellCentered)
      return RectDomain(BaseClass(lo(), hi() + iVec::unit(type)), dx, type, nGhost);
    return RectDomain(BaseClass(lo(), hi() - iVec::unit(staggered) + iVec::unit(type)),
                      dx, type, nGhost);
  }

protected:
  rVec dx;
  int staggered;
  int nGhost;
};

#endif // RECTGRID_H
