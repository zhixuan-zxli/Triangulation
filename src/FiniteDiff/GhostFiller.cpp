#include "Core/TensorSlice.h"
#include "FiniteDiff/GhostFiller.h"

template <int Dim>
template <class T>
void GhostFiller<Dim>::doFillGhosts(Tensor<Real, Dim> &aData, int D, int side, char type, const T &cData) const
{
  const int nG = rd.getNumGhost();
  assert(nG >= 1);
  int bound[] = {rd.lo()[D], rd.hi()[D]};
  if(side > 0)
    std::swap(bound[0], bound[1]);
  int ifrom, ito;

  if(type == 'P') {
    if(rd.getCentering() == D) {
      ito = bound[0] + side;
      ifrom = bound[1] + side;
    } else { // non-staggered in dimension D
      ito = bound[0] + side;
      ifrom = bound[1];
    }
    for(int i = 0; i < nG; ++i) {
      aData.slice(D, ito) = aData.slice(D, ifrom);
      ito += side;
      ifrom += side;
    }
  } else {
    auto dx = rd.spacing();
    if(rd.getCentering() == D) {
      if(type == 'N') {
        ito = bound[0] + side;
        aData.slice(D, ito) = aData.slice(D, ito - 2*side) + cData * (2.0 * dx[D]);
      } else if(type == 'D') {
        // set to the Dirichlet values and done
        aData.slice(D, bound[0]) = cData;
      }
    } else { // non-staggered dimension
      ito = bound[0] + side;
      if(type == 'N') {
        aData.slice(D, ito) = aData.slice(D, ito - side) + cData * dx[D];
      } else if(type == 'D') {
        aData.slice(D, ito) = aData.slice(D, ito - side) * (-1.0) + cData * 2.0;
      }
    }
  } // end if type == 'D' or 'N'
}

template <int Dim>
void GhostFiller<Dim>::fillGhosts(Tensor<Real, Dim> &aData, int D, int side, char type, const Tensor<Real, Dim - 1> &cData) const
{
  doFillGhosts(aData, D, side, type, cData);
}

template <int Dim>
void GhostFiller<Dim>::fillGhosts(Tensor<Real, Dim> &aData, int D, int side, char type, int) const
{
  doFillGhosts(aData, D, side, type, 0.0);
}

template <int Dim>
void GhostFiller<Dim>::fillGhosts(Tensor<Real, Dim> &aData, const char *types, const Tensor<Real, Dim - 1> *cData) const
{
  for(int d=0; d<Dim; ++d) {
    for(int side : {-1, 1}) {
      fillGhosts(aData, d, side, *types, *cData);
      ++types;
      ++cData;
    }
  }
}

template <int Dim>
void GhostFiller<Dim>::fillGhosts(Tensor<Real, Dim> &aData, const char *types, int) const
{
  for(int d=0; d<Dim; ++d) {
    for(int side : {-1, 1}) {
      fillGhosts(aData, d, side, *types, 0.0);
      ++types;
    }
  }
}

template <int Dim>
void GhostFiller<Dim>::copyGhosts(const Tensor<Real, Dim> &aSrc, Tensor<Real, Dim> &aDest) const
{
  const int nG = rd.getNumGhost();
  for(int D = 0; D < Dim; ++D) {
    for(int side : {-1, 1}) {
      const int bound = (side == -1) ? (rd.lo()[D]) : (rd.hi()[D]);
      for(int i = 1; i <= nG; ++i)
        aDest.slice(D, bound + i * side) = aSrc.slice(D, bound + i * side);
    }
  }
}

//============================================================
template class GhostFiller<SpaceDim>;
