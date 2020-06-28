#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include "Core/TensorSlicing.h"
#include "FiniteDiff/RectDomain.h"
#include "FiniteDiff/FuncFiller.h"
#include "FiniteDiff/GhostFiller.h"
#include "FiniteDiff/LevelOp.h"
#include "../example/common/TestUtility.h"

const int Dim = SpaceDim;
using rVec = Vec<Real, Dim>;
using iVec = Vec<int, Dim>;

template class Tensor<Real, Dim>;

#if DIM == 2

Real PHI(const rVec &x) {
  return sin(2*M_PI*x[0]) * cos(2*M_PI*x[1]);
}

Real LAPPHI(const rVec &x) {
  return -8*M_PI*M_PI * sin(2*M_PI*x[0]) * cos(2*M_PI*x[1]);
}

rVec VEL(const rVec &x) {
  return rVec { ipow<2>(sin(M_PI*x[0])) * sin(2*M_PI*x[1]),
                -sin(2*M_PI*x[0]) * ipow<2>(sin(M_PI*x[1])) };
}

rVec CNV(const rVec &x) {
  return rVec { ipow<2>(sin(M_PI*x[0])) * sin(2*M_PI*x[0]) * ipow<2>(sin(M_PI*x[1])),
                ipow<2>(sin(M_PI*x[1])) * sin(2*M_PI*x[1]) * ipow<2>(sin(M_PI*x[0])) } * (2.0*M_PI);
}

#else

Real PHI(const rVec &x) {
  return sin(2 * M_PI * x[0]) * sin(2 * M_PI * x[1]) * cos(2 * M_PI * x[2]);
}

Real LAPPHI(const rVec &x) {
  return -12.0 * (M_PI * M_PI) * sin(2 * M_PI * x[0]) * sin(2 * M_PI * x[1]) * cos(2 * M_PI * x[2]);
}

rVec VEL(const rVec &x) {
  return rVec { ipow<2>(sin(M_PI*x[0])) * sin(2*M_PI*x[1]) * sin(2*M_PI*x[2]) * (1.0/2),
                sin(2*M_PI*x[0]) * ipow<2>(sin(M_PI*x[1])) * sin(2*M_PI*x[2]) * (1.0/2),
                sin(2*M_PI*x[0]) * sin(2*M_PI*x[1]) * ipow<2>(sin(M_PI*x[2])) * (-1.0) };
}

rVec CNV(const rVec &pos) {
  const Real x = pos[0], y = pos[1], z = pos[2];
  Real r0 = (M_PI/2) * ipow<2>(sin(M_PI*x)) * (cos(M_PI*x) * sin(M_PI*x) * ipow<2>(sin(2*M_PI*y)) * ipow<2>(sin(2*M_PI*z))
      + cos(2*M_PI*y) * sin(2*M_PI*x) * ipow<2>(sin(M_PI*y)) * ipow<2>(sin(2*M_PI*z))
      - 2 * cos(2*M_PI*z) * sin(2*M_PI*x) * ipow<2>(sin(2*M_PI*y)) * ipow<2>(sin(M_PI*z)));
  Real r1 = (M_PI/2) * ipow<2>(sin(M_PI*y)) * (cos(2*M_PI*x) * ipow<2>(sin(M_PI*x)) * sin(2*M_PI*y) * ipow<2>(sin(2*M_PI*z))
      + cos(M_PI*y) * ipow<2>(sin(2*M_PI*x)) * sin(M_PI*y) * ipow<2>(sin(2*M_PI*z))
      - 2 * cos(2*M_PI*z) * ipow<2>(sin(2*M_PI*x)) * sin(2*M_PI*y) * ipow<2>(sin(M_PI*z)));
  Real r2 = (-M_PI) * ipow<2>(sin(M_PI*z)) * (cos(2*M_PI*x) * ipow<2>(sin(M_PI*x)) * ipow<2>(sin(2*M_PI*y)) * sin(2*M_PI*z)
      + cos(2*M_PI*y) * ipow<2>(sin(2*M_PI*x)) * ipow<2>(sin(M_PI*y)) * sin(2*M_PI*z)
      - 2 * cos(M_PI*z) * ipow<2>(sin(2*M_PI*x)) * ipow<2>(sin(2*M_PI*y)) * sin(M_PI*z));
  return rVec {r0, r1, r2};
}

#endif

const int numRatesOfOutput = 12;
const bool testLaplacian = true;
const bool testLaplacianWithGhosts = true;
const bool testFilter = true;
const bool testConvection = true;

std::array<Real,numRatesOfOutput> testOps(int N)
{
  std::array<Real, numRatesOfOutput> errnorm;
  RectDomain<Dim> rd(Box<Dim>(0, N-1), 1.0/N, RectDomain<Dim>::CellCentered, 2);
  FuncFiller<Dim> ff(rd);
  GhostFiller<Dim> gf(rd);
  LevelOp<Dim> fvop(rd);
  Box<Dim> gbx = rd.getGhostedBox();
  Tensor<Real, Dim> phi(gbx);
  Tensor<Real, Dim> err(gbx);
  Tensor<Real, Dim> res(gbx);
  Tensor<Real, Dim-1> bcData[2*Dim];
  for(int k = 0; k < 2*Dim; ++k)
    bcData[k].resize(reduce(gbx, k/2));
  // test Laplacian
  if(testLaplacian) {
    ff.fillAvr(phi, RectDomain<Dim>::CellCentered, PHI);
    ff.fillAvr(err, RectDomain<Dim>::CellCentered, LAPPHI);
    fvop.computeLaplacian(phi, res);
    err = err - res;
    for (int p = 0; p <= 2; ++p)
      errnorm[p] = fvop.computeNorm(err, RectDomain<Dim>::CellCentered, p);
  }
  // test Laplacian with ghosts
  if(testLaplacianWithGhosts) {
    ff.fillAvr(phi, RectDomain<Dim>::CellCentered, PHI);
    ff.fillAvr(err, RectDomain<Dim>::CellCentered, LAPPHI);
    ff.fillBdryAvr(bcData, RectDomain<Dim>::CellCentered, PHI);
    gf.fillGhosts(phi, "DDDDDD", bcData);
    fvop.computeLaplacian(phi, res);
    err = err - res;
    for (int p = 0; p <= 2; ++p)
      errnorm[3 + p] = fvop.computeNorm(err, RectDomain<Dim>::CellCentered, p);
  }
  // test filter
  RectDomain<Dim> uGrid[Dim];
  Box<Dim> ugbx[Dim];
  Tensor<Real, Dim+1> cVel(pGrid, Dim);
  Tensor<Real, Dim+1> cErr[Dim];
  Tensor<Real, Dim> fVel[Dim];
  for (int d = 0; d < Dim; ++d) {
    uGrid[d] = rd.stagger(d);
    ugbx[d] = uGrid[d].getGhostedBox();
    cVel[d].resize(gbx);
    cErr[d].resize(gbx);
    fVel[d].resize(ugbx[d]);
  }
  if(testFilter) {
    for(int d = 0; d < Dim; ++d) {
      ff.fillAvr(cErr[d], RectDomain<Dim>::CellCentered, [&](const rVec &x) { return VEL(x)[d]; });
      ff.fillAvr(fVel[d], d, [&](const rVec &x) { return VEL(x)[d]; });
    }
    fvop.filterFace2Cell(fVel, cVel);
    for (int d = 0; d < Dim; ++d)
      cErr[d] = cErr[d] - cVel[d];
    for (int p = 0; p <= 2; ++p) {
      rVec errComp;
      for (int d = 0; d < Dim; ++d)
        errComp[d] = fvop.computeNorm(cErr[d], RectDomain<Dim>::CellCentered, p);
      errnorm[6 + p] = norm(errComp, p);
    }
  }
  // test convection
  if(testConvection) {
    Tensor<Real, Dim> fCnv[Dim];
    Tensor<Real, Dim> fErr[Dim];
    for (int d = 0; d < Dim; ++d) {
      fCnv[d].resize(ugbx[d]);
      fErr[d].resize(ugbx[d]);
      ff.fillAvr(fErr[d], d, [&](const rVec &x) { return CNV(x)[d]; });
//      fErr[d] = 0.0;
      ff.fillAvr(fVel[d], d, [&](const rVec &x) { return VEL(x)[d]; });
//      fVel[d] = 0.0;
    }
//    for(int n = 0; n < 100; ++n)
    fvop.computeConvection(fVel, fCnv);
    for (int d = 0; d < Dim; ++d) {
      fErr[d] = fErr[d] - fCnv[d];
      fErr[d].slice(d, 0) = 0.0;
      fErr[d].slice(d, N) = 0.0;
    }
    for (int p = 0; p <= 2; ++p) {
      rVec errComp;
      for (int d = 0; d < Dim; ++d)
        errComp[d] = fvop.computeNorm(fErr[d], d, p);
      errnorm[9 + p] = norm(errComp, p);
    }
  }
  return errnorm;
}

void doTest(const std::vector<int> &gridSize)
{
  const int numGrid = gridSize.size();
  std::vector<std::array<Real, numRatesOfOutput>> errnorm(numGrid);
  for(int n = numGrid-1; n >= 0; --n) {
    std::cout << "\nTesting 1/h = " << gridSize[n] << std::endl;
    errnorm[n] = testOps(gridSize[n]);
  }
  printConvergenceTable(&gridSize[0], errnorm);
}

int main()
{
  reset_dbglevel(1);
  std::cout << "SpaceDim = " << SpaceDim << std::endl;
  std::cout << "===================================================================================" << std::endl;
#if DIM == 2
  doTest({256, 128, 64});
#else
  doTest({128, 64, 32});
#endif // DIM == 2
  return 0;
}
