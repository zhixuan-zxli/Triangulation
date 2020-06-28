#ifndef FUNCFILLER_H
#define FUNCFILLER_H

#include "Core/numlib.h"
#include "Core/Tensor.h"
#include "FiniteDiff/RectDomain.h"

template <int Dim>
class FuncFiller
{
public:
  using iVec = Vec<int, Dim>;
  using rVec = Vec<Real, Dim>;

  // Always initialize a FuncFiller with a p-grid.
  FuncFiller(const RectDomain<Dim> &_rd) : rd(_rd) { }

public:
  template <class TFunc>
  void fillPoint(Tensor<Real, Dim> &aData, int staggered, const TFunc &expr, bool fillGhosts = true) const;

  template <class TFunc>
  void fillAvr(Tensor<Real,Dim> &aData, int d, const TFunc &expr, bool fillGhosts = true) const;

  template <class TFunc>
  void fillBdryAvr(Tensor<Real,Dim-1> &aData, int staggered, int D, int side, const TFunc &expr, bool fillGhosts = true) const;

  template <class TFunc>
  void fillBdryAvr(Tensor<Real,Dim-1> *aData, int staggered, const TFunc &expr, bool fillGhosts = true) const;

protected:
  const RectDomain<Dim> rd;
};

//============================================================

template <int Dim>
template <class TFunc>
inline
void FuncFiller<Dim>::fillPoint(Tensor<Real, Dim> &aData, int staggered, const TFunc &expr, bool fillGhosts) const
{
  RectDomain<Dim> srd = rd.stagger(staggered);
  rVec dx = rd.spacing();
  rVec delta = rd.getDelta();
  Box<Dim> bx = srd;
  if(fillGhosts)
    bx = srd.getGhostedBox();
  assert(aData.box().contain(bx));
  ddfor(bx, [&](const iVec &idx) { aData(idx) = expr((idx + delta) * dx); });
}

template <>
template <class TFunc>
inline
void FuncFiller<2>::fillAvr(Tensor<Real, 2> &aData, int d, const TFunc &expr, bool fillGhosts) const
{
  RectDomain<2> srd = rd.stagger(d);
  rVec dx = rd.spacing();
  Box<2> bx = srd;
  if(fillGhosts)
    bx = srd.getGhostedBox();
  assert(aData.box().contain(bx));
  //
  if(d == RectDomain<2>::CellCentered) { // cell-average version
    Real cellVol = prod(dx);
#pragma omp parallel for default(shared) schedule(static)
    loop_box_2(bx, i, j) {
      iVec idx {i, j};
      aData(i, j) = quad<4>(expr, idx*dx, (idx+1)*dx) / cellVol;
    }
  } else { // face-average version
    Real faceVol = prod(reduce(dx, d));
#pragma omp parallel for default(shared) schedule(static)
    loop_box_2(bx, i, j) {
      iVec idx {i, j};
      rVec corners[2] {idx*dx, (idx+1)*dx};
      aData(i, j) = quad<4>([&](const Vec<Real, 1> &x) { return expr(enlarge(x, corners[0][d], d)); },
                            reduce(corners[0], d), reduce(corners[1], d)) / faceVol;
    }
  }
}

template <>
template <class TFunc>
inline
void FuncFiller<2>::fillBdryAvr(Tensor<Real, 1> &aData,
                                int staggered,
                                int D,
                                int side,
                                const TFunc &expr,
                                bool fillGhosts) const
{
  RectDomain<2> srd = rd.stagger(staggered);
  auto dx = rd.spacing();
  auto dxd = reduce(dx, D);
  auto bx = reduce(srd, D);
  if(fillGhosts)
    bx = bx.inflate(rd.getNumGhost());
  assert(aData.box().contain(bx));
  Real hat = ((side < 0) ? (rd.lo()[D]) : (rd.hi()[D]+1)) * dx[D];
  if(staggered == RectDomain<2>::CellCentered || staggered == D) { // a one-dimension reduction
    Real faceVol = dxd[0];
#pragma omp parallel for default(shared) schedule(static)
    loop_box_1(bx, i) {
      Vec<Real, 1> corners[2] { dxd*i, dxd*(i+1) };
      aData(i) = quad<4>([&](const Vec<Real, 1> &x) { return expr(enlarge(x, hat, D)); }, corners[0], corners[1]) / faceVol;
    }
  } else {
#pragma omp parallel for default(shared) schedule(static)
    loop_box_1(bx, i) {
      aData(i) = expr(enlarge(dxd*i, hat, D));
    }
  }
}

template <>
template <class TFunc>
inline
void FuncFiller<3>::fillAvr(Tensor<Real, 3> &aData, int d, const TFunc &expr, bool fillGhosts) const
{
  RectDomain<3> srd = rd.stagger(d);
  rVec dx = rd.spacing();
  Box<3> bx = srd;
  if(fillGhosts)
    bx = srd.getGhostedBox();
  assert(aData.box().contain(bx));
  //
  if(d == RectDomain<3>::CellCentered) { // cell-average version
    Real cellVol = prod(dx);
#pragma omp parallel for default(shared) schedule(static)
    loop_box_3(bx, i, j, k) {
      iVec idx {i, j, k};
      aData(i, j, k) = quad<4>(expr, idx*dx, (idx+1)*dx) / cellVol;
    }
  } else { // face-average version
    Real faceVol = prod(reduce(dx, d));
#pragma omp parallel for default(shared) schedule(static)
    loop_box_3(bx, i, j, k) {
      iVec idx {i, j, k};
      rVec corners[2] {idx*dx, (idx+1)*dx};
      aData(i, j, k) = quad<4>([&](const Vec<Real, 2> &x) { return expr(enlarge(x, corners[0][d], d)); },
                               reduce(corners[0], d), reduce(corners[1], d)) / faceVol;
    }
  }
}

template <>
template <class TFunc>
inline
void FuncFiller<3>::fillBdryAvr(Tensor<Real, 2> &aData,
                                int staggered,
                                int D,
                                int side,
                                const TFunc &expr,
                                bool fillGhosts) const
{
  RectDomain<3> srd = rd.stagger(staggered);
  auto dx = rd.spacing();
  auto dxd = reduce(dx, D);
  auto bx = reduce(srd, D);
  if(fillGhosts)
    bx = bx.inflate(rd.getNumGhost());
  assert(aData.box().contain(bx));
  Real hat = ((side < 0) ? (rd.lo()[D]) : (rd.hi()[D]+1)) * dx[D];
  if(staggered == RectDomain<3>::CellCentered || staggered == D) { // a one-dimension reduction
    Real faceVol = prod(dxd);
#pragma omp parallel for default(shared) schedule(static)
    loop_box_2(bx, i, j) {
      Vec<int, 2> idx {i, j};
      Vec<Real, 2> corners[2] { idx*dxd, (idx+1)*dxd };
      aData(i, j) = quad<4>([&](const Vec<Real, 2> &x) { return expr(enlarge(x, hat, D)); }, corners[0], corners[1]) / faceVol;
    }
  } else { // a two-dimension reduction
    if(staggered > D)
      --staggered;
    Real edgeVol = reduce(dxd, staggered)[0];
#pragma omp parallel for default(shared) schedule(static)
    loop_box_2(bx, i, j) {
      Vec<int, 2> idx {i, j};
      Vec<Real, 2> corners[2] {idx * dxd, (idx+1) * dxd};
      aData(i, j) = quad<4>([&](const Vec<Real, 1> &x) {
        return expr(enlarge(enlarge(x, corners[0][staggered], staggered), hat, D));
      }, reduce(corners[0], staggered), reduce(corners[1], staggered)) / edgeVol;
    }
  }
}

template <int Dim>
template <class TFunc>
inline
void FuncFiller<Dim>::fillBdryAvr(Tensor<Real, Dim-1> *aData, int staggered, const TFunc &expr, bool fillGhosts) const
{
  int k = 0;
  for(int d = 0; d < Dim; ++d)
    for(int side : {-1, 1})
      fillBdryAvr(aData[k++], staggered, d, side, expr, fillGhosts);
}

#endif //FUNCFILLER_H
