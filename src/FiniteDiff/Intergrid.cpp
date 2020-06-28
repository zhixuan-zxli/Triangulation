#include "FiniteDiff/Intergrid.h"

template <int d>
void applyToFace(const Tensor<Real, 2> &fineData, Tensor<Real, 2> &coarseData, const RectDomain<2> &coarseGrid)
{
  constexpr int em[] = {d==1, d==0};
  Box<2> sbx = coarseGrid.stagger(d);
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(sbx, i, j) {
    coarseData(i, j) = (fineData(2*i, 2*j) + fineData(2*i+em[0], 2*j+em[1])) / 2;
  }
}

template <>
void Intergrid<2>::applyRestriction(const Tensor<Real, 2> &fineData, Tensor<Real, 2> &coarseData, int d) const
{
  if(d == RectDomain<2>::CellCentered) {
#pragma omp parallel for default(shared) schedule(static)
    loop_box_2(coarseGrid, i, j) {
        coarseData(i, j) = (fineData(2 * i, 2 * j) + fineData(2 * i + 1, 2 * j) + fineData(2 * i, 2 * j + 1)
            + fineData(2 * i + 1, 2 * j + 1)) / 4;
      }
  } else if(d == 0) {
    applyToFace<0>(fineData, coarseData, coarseGrid);
  } else if(d == 1) {
    applyToFace<1>(fineData, coarseData, coarseGrid);
  }
}

template <int d>
void applyToFace(const Tensor<Real, 3> &fineData, Tensor<Real, 3> &coarseData, const RectDomain<3> &coarseGrid)
{
  constexpr int em[] = {d==2, d==0, d==1};
  constexpr int el[] = {d==1, d==2, d==0};
  Box<3> sbx = coarseGrid.stagger(d);
#pragma omp parallel for default(shared) schedule(static)
  loop_box_3(sbx, i, j, k) {
    coarseData(i,j,k) = (fineData(2*i, 2*j, 2*k) + fineData(2*i+em[0], 2*j+em[1], 2*k+em[2])
        + fineData(2*i+el[0], 2*j+el[1], 2*k+el[2]) + fineData(2*i+em[0]+el[0], 2*j+em[1]+el[1], 2*k+em[2]+el[2])) / 4;
  }
}

template <>
void Intergrid<3>::applyRestriction(const Tensor<Real, 3> &fineData, Tensor<Real, 3> &coarseData, int d) const
{
  if(d == RectDomain<3>::CellCentered) {
#pragma omp parallel for default(shared) schedule(static)
    loop_box_3(coarseGrid, i, j, k) {
          coarseData(i, j, k) = (fineData(2 * i, 2 * j, 2 * k) + fineData(2 * i + 1, 2 * j, 2 * k)
              + fineData(2 * i, 2 * j + 1, 2 * k) + fineData(2 * i + 1, 2 * j + 1, 2 * k)
              + fineData(2 * i, 2 * j, 2 * k + 1) + fineData(2 * i + 1, 2 * j, 2 * k + 1)
              + fineData(2 * i, 2 * j + 1, 2 * k + 1) + fineData(2 * i + 1, 2 * j + 1, 2 * k + 1)) / 8;
        }
  } else if(d == 0) {
    applyToFace<0>(fineData, coarseData, coarseGrid);
  } else if(d == 1) {
    applyToFace<1>(fineData, coarseData, coarseGrid);
  } else if(d == 2) {
    applyToFace<2>(fineData, coarseData, coarseGrid);
  }
}

template <>
void Intergrid<2>::incrementalProlong(const Tensor<Real, 2> &coarseData, Tensor<Real, 2> &fineData) const
{
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(fineGrid, i, j) {
    fineData(i, j) += coarseData(i/2, j/2);
  }
}

template <>
void Intergrid<3>::incrementalProlong(const Tensor<Real, 3> &coarseData, Tensor<Real, 3> &fineData) const
{
#pragma omp parallel for default(shared) schedule(static)
  loop_box_3(fineGrid, i, j, k) {
      fineData(i, j, k) += coarseData(i/2, j/2, k/2);
    }
}
