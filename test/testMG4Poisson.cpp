#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cstring>
#include "Core/TensorExpr.h"
#include "FiniteDiff/FuncFiller.h"
#include "FiniteDiff/GhostFiller.h"
#include "FiniteDiff/LevelOp.h"
#include "FiniteDiff/Intergrid.h"
#include "FiniteDiff/Multigrid.h"
#include "../example/common/TestUtility.h"

const int Dim = SpaceDim;
using rVec = Vec<Real,Dim>;
using iVec = Vec<int, Dim>;

template class Tensor<Real,Dim>;

#if DIM == 2
Real PHI(const rVec &x) {
  return sin(2 * M_PI * x[0]) * cos(2 * M_PI * x[1]);
}
rVec DPHI(const rVec &x) {
  return rVec { 2*M_PI * cos(2*M_PI*x[0]) * cos(2*M_PI*x[1]),
                -2*M_PI * sin(2*M_PI*x[0]) * sin(2*M_PI*x[1]) };
}
Real D2PHI(const rVec &x) {
  return -8.0 * (M_PI * M_PI) * sin(2 * M_PI * x[0]) * cos(2 * M_PI * x[1]);
}
// (*) If you want to compare the performance between this code and the Fortran code,
// use the following solution that has a primitive.
/*
Real I_PHI(const rVec &x, Real h) {
  auto i_sin = [&](Real omega, Real a) { return 2.0 / (omega*h) * sin(omega*h/2) * sin(omega*a); };
  return i_sin(2*M_PI, x[0]) + i_sin(4*M_PI, x[0])
      + i_sin(2*M_PI, x[1]) + i_sin(4*M_PI, x[1]);
}
Real I_D2PHI(const rVec &x, Real h) {
  auto i_sin = [&](Real omega, Real a) { return 2.0 / (omega*h) * sin(omega*h/2) * sin(omega*a); };
  return (-4*M_PI*M_PI) * i_sin(2*M_PI, x[0]) + (-16*M_PI*M_PI) * i_sin(4*M_PI, x[0])
      + (-4*M_PI*M_PI) * i_sin(2*M_PI, x[1]) + (-16*M_PI*M_PI) * i_sin(4*M_PI, x[1]);
}
 */
#else
Real PHI(const rVec &x) {
  return sin(2 * M_PI * x[0]) * sin(2 * M_PI * x[1]) * cos(2 * M_PI * x[2]);
}
rVec DPHI(const rVec &x) {
  return rVec { 2*M_PI * cos(2*M_PI*x[0]) * sin(2*M_PI*x[1]) * cos(2*M_PI*x[2]),
                2*M_PI * sin(2*M_PI*x[0]) * cos(2*M_PI*x[1]) * cos(2*M_PI*x[2]),
                -2*M_PI * sin(2*M_PI*x[0]) * sin(2*M_PI*x[1]) * sin(2*M_PI*x[2]) };
}
Real D2PHI(const rVec &x) {
  return -12.0 * (M_PI * M_PI) * sin(2 * M_PI * x[0]) * sin(2 * M_PI * x[1]) * cos(2 * M_PI * x[2]);
}
#endif

std::array<Real, 3> testMG(const std::vector<const MGLevelOp<Dim> *> &opHier,
                           const std::vector<const Intergrid<Dim> *> &trHier,
                           const char *bcTypes)
{
  const RectDomain<Dim> rd = opHier[0]->getDomain();
  FuncFiller<Dim> ff(rd);
  LevelOp<Dim> fvOp(rd);
  Box<Dim> gbx = rd.getGhostedBox();
  Tensor<Real, Dim> phi(gbx);
  Tensor<Real, Dim> err(gbx);
  Tensor<Real, Dim> rhs(gbx);
  Tensor<Real, Dim-1> bcData[2*Dim];
  int k = 0;
  for(int d = 0; d < Dim; ++d) {
    for(int side : {-1, 1}) {
      bcData[k].resize(reduce(gbx, d));
      if(bcTypes[k] == 'D') {
        ff.fillBdryAvr(bcData[k], RectDomain<Dim>::CellCentered, d, side, PHI);
      } else if(bcTypes[k] == 'N') {
        ff.fillBdryAvr(bcData[k], RectDomain<Dim>::CellCentered, d, side,
                       [&](const rVec &x) { return side * DPHI(x)[d]; });
      }
      ++k;
    }
  }
  ff.fillAvr(err, RectDomain<Dim>::CellCentered, PHI);
  ff.fillAvr(rhs, RectDomain<Dim>::CellCentered, D2PHI);
  // (*) In case that you want to compare the performance between this code and the Fortran code.
//  auto dx = rd.spacing();
//  ff.fillPoint(err, RectDomain<Dim>::CellCentered, [&](const rVec &x) { return I_PHI(x, dx[0]); });
//  ff.fillPoint(rhs, RectDomain<Dim>::CellCentered, [&](const rVec &x) { return I_D2PHI(x, dx[0]); });
  phi = 0.0;

  Multigrid<Dim> mg(opHier, trHier);
  mg.setParam({2, 2, 20}, 20, 1.001, 1e-10);
  push_dbglevel(1);
  mg.solve(phi, rhs, bcData);
  pop_dbglevel();

  if(strchr(bcTypes, 'D') == nullptr)
    err = err - phi - (err(0,0) - phi(0,0));
  else
    err = err - phi;
  std::array<Real, 3> errnorm;
  for(int p = 0; p <= 2; ++p)
    errnorm[p] = fvOp.computeNorm(err, RectDomain<Dim>::CellCentered, p);
  return errnorm;
}

int main()
{
  std::cout << "SpaceDim = " << SpaceDim << std::endl;
  std::cout << "===================================================================================" << std::endl;
#if DIM == 2
  const int gridSize[] = {/*4096, 2048, 1024, 512, */256, 128, 64, 32, 16, 8};
#else
  const int gridSize[] = {128, 64, 32, 16, 8};
#endif // DIM == 2
  const int numGrid = std::extent<decltype(gridSize)>::value;
  const int numCompHier = 3;
  char bcTypes[] = "NNNNNN";
  // (*) In case that you want to compare the performance between this code and the Fortran code.
//  char bcTypes[] = "PPPPPP";
  bcTypes[2*Dim] = '\0';
  std::vector<const RectDomain<Dim> *> gridHier;
  std::vector<const MGLevelOp<Dim> *> opHier;
  std::vector<const Intergrid<Dim> *> trHier;
  for(int n = 0; n < numGrid; ++n) {
    const int N = gridSize[n];
    const RectDomain<Dim> *pRd = new RectDomain<Dim>(Box<Dim>(0, N-1), 1.0/N, RectDomain<Dim>::CellCentered, 2);
    gridHier.push_back(pRd);
    const MGLevelOp<Dim> *plv = new MGLevelOp<Dim>(*pRd, 0.8, bcTypes);
    opHier.push_back(plv);
    if(n != 0) {
      const Intergrid<Dim> *pt = new Intergrid<Dim>(*pRd, *(gridHier[n-1]));
      trHier.push_back(pt);
    }
  }

  std::vector<std::array<Real, 3>> errnorm(numCompHier);
  for(int n = numCompHier-1; n >= 0; --n) {
    std::cout << "\nTesting 1/h = " << gridSize[n] << std::endl;
    std::vector<const MGLevelOp<Dim> *> _opHier(opHier.cbegin() + n, opHier.cend());
    std::vector<const Intergrid<Dim> *> _trHier(trHier.cbegin() + n, trHier.cend());
    errnorm[n] = testMG(_opHier, _trHier, bcTypes);
  }

  for(int n = numGrid-1; n >= 0; --n) {
    if(n != 0)
      delete trHier[n-1];
    delete opHier[n];
    delete gridHier[n];
  }

  printConvergenceTable(&gridSize[0], errnorm);
  return 0;
}