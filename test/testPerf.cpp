#include <iostream>
#include <iomanip>
#include "Core/Wrapper_LAPACKE.h"
#include "Core/TensorExpr.h"
#include "FiniteDiff/RectDomain.h"
#include "FiniteDiff/LevelOp.h"
#include "../example/common/TestUtility.h"

const int Dim = SpaceDim;

extern "C" void relax2d_(Real *phi, Real *output, Real *rhs,
                         int *m, int *n,
                         Real *dx, Real *dy, Real *w);

void testRelaxJacobi(int N, int iter, bool useFortran)
{
  RectDomain<Dim> rd(Box<Dim>(0, N-1), 1.0/N, RectDomain<Dim>::CellCentered, 1);
  const auto gbx = rd.getGhostedBox();
  Tensor<Real, Dim> phi[2], rhs;
  for(int i = 0; i < 2; ++i) {
    phi[i].resize(gbx);
    phi[i] = 0.0;
  }
  rhs.resize(gbx);
  rhs = 1.0;
  Real w = 0.6;
  if(useFortran) {
    int m[] = { N+2, N+2 };
    Real h[] = { 1.0/N, 1.0/N };
    relax2d_(phi[0].data(), phi[1].data(), rhs.data(), &m[0], &m[1], &h[0], &h[1], &w); // warm up
    CPUTimer timer;
    for(int k = 0; k < iter; ++k) {
      relax2d_(phi[k%2].data(), phi[(k+1)%2].data(), rhs.data(), &m[0], &m[1], &h[0], &h[1], &w);
    }
    std::cout << std::setw(16) << "Fortran : " << timer() << " s" << std::endl;
  } else {
    LevelOp<Dim> lv(rd);
    lv.relaxJacobi(phi[0], rhs, phi[1], w); // warm up
    CPUTimer timer;
    for(int k = 0; k < iter; ++k) {
      lv.relaxJacobi(phi[k%2], rhs, phi[(k+1)%2], w);
    }
    std::cout << std::setw(16) << "C++ Tensor : " << timer() << " s" << std::endl;
  }
}

void test2WayAddition(int N, int iter, bool useFortran)
{
  Tensor<Real,1> a(N), b(N);
  for(int i = 0; i < N; ++i) {
    a(i) = i;
    b(i) = 2*i;
  }
  if(useFortran) {
    cblas_daxpy(N, 2.0, a.data(), 1, b.data(), 1); // warm up
    CPUTimer timer;
    for(int k = 0; k < iter; ++k)
      cblas_daxpy(N, 2.0, a.data(), 1, b.data(), 1);
    std::cout << std::setw(16) << "Fortran : " << timer() << " s" << std::endl;
  } else {
    b = b + a * 2.0; // warm up
    CPUTimer timer;
    for(int k = 0; k < iter; ++k)
      b = b + a * 2.0;
    std::cout << std::setw(16) << "C++ Tensor : " << timer() << " s" << std::endl;
  }
}

void test3WayAddition(int N, int iter, bool useFortran)
{
  Tensor<Real,1> a(N), b(N), c(N);
  for(int i = 0; i < N; ++i) {
    a(i) = i;
    b(i) = 2*i;
    c(i) = -i;
  }
  if(useFortran) {
    cblas_daxpy(N, 2.0, a.data(), 1, b.data(), 1); // warm up
    cblas_daxpy(N, -1.0, c.data(), 1, b.data(), 1); // warm up
    CPUTimer timer;
    for(int k = 0; k < iter; ++k) {
      cblas_daxpy(N, 2.0, a.data(), 1, b.data(), 1);
      cblas_daxpy(N, -1.0, c.data(), 1, b.data(), 1);
    }
    std::cout << std::setw(16) << "Fortran : " << timer() << " s" << std::endl;
  } else {
    b = b + a * 2.0 - c * 1.0; // warm up
    CPUTimer timer;
    for(int k = 0; k < iter; ++k)
      b = b + a * 2.0 - c * 1.0;
    std::cout << std::setw(16) << "C++ Tensor : " << timer() << " s" << std::endl;
  }
}

int main(int argc, char *argv[])
{
  std::cout << "Test Jacobi relaxation : " << std::endl;
  testRelaxJacobi(1024, 100, true);
  testRelaxJacobi(1024, 100, false);
  std::cout << "Test 2-way addition : " << std::endl;
  test2WayAddition(1 << 24, 10, true);
  test2WayAddition(1 << 24, 10, false);
  std::cout << "Test 3-way addition : " << std::endl;
  test3WayAddition(1 << 24, 10, true);
  test3WayAddition(1 << 24, 10, false);
  return 0;
}