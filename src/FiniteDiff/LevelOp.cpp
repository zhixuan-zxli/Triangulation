#include "FiniteDiff/LevelOp.h"
#include "Core/TensorSlice.h"
#include "Core/numlib.h"

template <>
template <class T>
void LevelOp<2>::computeLaplacian(Real alpha, const Tensor<Real, 2> &phi, const T &rhs, Tensor<Real, 2> &LofPhi) const
{
  auto dx = rd.spacing();
  Box<2> valid = rd;
  assert(LofPhi.box().contain(valid));
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(valid, i, j) {
      Real r = (phi(i-1,j) - phi(i,j)*2 + phi(i+1,j)) / (dx[0]*dx[0])
          + (phi(i,j-1) - phi(i,j)*2 + phi(i,j+1)) / (dx[1]*dx[1]);
      LofPhi(i,j) = alpha * r + rhs(i,j);
    }
}

template <>
template <class T>
void LevelOp<3>::computeLaplacian(Real alpha, const Tensor<Real, 3> &phi, const T &rhs, Tensor<Real, 3> &LofPhi) const
{
  auto dx = rd.spacing();
  Box<3> valid = rd;
  assert(LofPhi.box().contain(valid));
#pragma omp parallel for default(shared) schedule(static)
  loop_box_3(valid, i, j, k) {
      Real r = (phi(i-1,j,k) - phi(i,j,k)*2 + phi(i+1,j,k)) / (dx[0]*dx[0])
          + (phi(i,j-1,k) - phi(i,j,k)*2 + phi(i,j+1,k)) / (dx[1]*dx[1])
          + (phi(i,j,k-1) - phi(i,j,k)*2 + phi(i,j,k+1)) / (dx[2]*dx[2]);
      LofPhi(i,j,k) = alpha * r + rhs(i,j,k);
    }
}

template <int Dim>
void LevelOp<Dim>::computeLaplacian(const Tensor<Real, Dim> &phi, Tensor<Real, Dim> &LofPhi) const
{
  computeLaplacian(1.0, phi, [](auto...) { return 0.0; }, LofPhi);
}

template <int Dim>
void LevelOp<Dim>::computeLaplacian(const Tensor<Real, Dim> &phi, const Tensor<Real, Dim> &rhs, Tensor<Real, Dim> &rsd) const
{
  computeLaplacian(-1.0, phi, rhs, rsd);
}

template <>
void LevelOp<2>::relaxJacobi(const Tensor<Real, 2> &phi, const Tensor<Real, 2> &rhs, Tensor<Real, 2> &JofPhi, Real w) const
{
  auto dx = rd.spacing();
  Box<2> valid = rd;
  assert(JofPhi.box().contain(valid));
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(valid, i, j) {
      Real a = (phi(i-1,j) + phi(i+1,j)) / (dx[0]*dx[0])
          + (phi(i,j-1) + phi(i,j+1)) / (dx[1]*dx[1]);
      Real b = 2.0 * (1.0/(dx[0]*dx[0]) + 1.0/(dx[1]*dx[1]));
      Real r = (a - rhs(i, j)) / b;
      JofPhi(i, j) = r * w + phi(i, j) * (1-w);
    }
}

template <>
void LevelOp<3>::relaxJacobi(const Tensor<Real, 3> &phi, const Tensor<Real, 3> &rhs, Tensor<Real, 3> &JofPhi, Real w) const
{
  auto dx = rd.spacing();
  Box<3> valid = rd;
  assert(JofPhi.box().contain(valid));
#pragma omp parallel for default(shared) schedule(static)
  loop_box_3(valid, i, j, k) {
      Real a = (phi(i-1,j,k) + phi(i+1,j,k)) / (dx[0]*dx[0])
          + (phi(i,j-1,k) + phi(i,j+1,k)) / (dx[1]*dx[1])
          + (phi(i,j,k-1) + phi(i,j,k+1)) / (dx[2]*dx[2]);
      Real b = 2.0 * (1.0/(dx[0]*dx[0]) + 1.0/(dx[1]*dx[1]) + 1.0/(dx[2]*dx[2]));
      Real r = (a - rhs(i, j, k)) / b;
      JofPhi(i, j, k) = r * w + phi(i, j, k) * (1-w);
    }
}

template <int Dim>
Real LevelOp<Dim>::computeNorm(const Tensor<Real, Dim> &lhs, int staggered, int nt) const
{
  Box<Dim> valid = rd.stagger(staggered);
  if(nt == 0)
    return norm(lhs.slice(valid), 0);
  else if(nt == 1)
    return norm(lhs.slice(valid), 1) * prod(rd.spacing());
  else if(nt == 2)
    return norm(lhs.slice(valid), 2) * sqrt(prod(rd.spacing()));
  return 0.0;
}

//================================================================================

template <>
void LevelOp<2>::computeDivergence(const Tensor<Real, 2> *f, Tensor<Real, 2> &divf) const
{
  Box<2> valid = rd;
  auto dx = rd.spacing();
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(valid, i, j) {
      divf(i, j) = (f[0](i+1,j) - f[0](i,j)) / dx[0]
          + (f[1](i,j+1) - f[1](i,j)) / dx[1];
    }
}

template <>
void LevelOp<3>::computeDivergence(const Tensor<Real, 3> *f, Tensor<Real, 3> &divf) const
{
  Box<3> valid = rd;
  auto dx = rd.spacing();
#pragma omp parallel for default(shared) schedule(static)
  loop_box_3(valid, i, j, k) {
      divf(i, j, k) = (f[0](i+1,j,k) - f[0](i,j,k)) / dx[0]
          + (f[1](i,j+1,k) - f[1](i,j,k)) / dx[1]
          + (f[2](i,j,k+1) - f[2](i,j,k)) / dx[2];
    }
}

template <>
void LevelOp<2>::computeGradient(const Tensor<Real, 2> &f, Tensor<Real, 2> *df) const
{
  auto dx = rd.spacing();
  Box<2> valid = rd.stagger(0);
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(valid, i, j)
    df[0](i,j) = (f(i,j) - f(i-1,j)) / dx[0];
  valid = rd.stagger(1);
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(valid, i, j)
    df[1](i,j) = (f(i,j) - f(i,j-1)) / dx[1];
}

template <>
void LevelOp<3>::computeGradient(const Tensor<Real, 3> &f, Tensor<Real, 3> *df) const
{
  auto dx = rd.spacing();
  Box<3> valid = rd.stagger(0);
#pragma omp parallel for default(shared) schedule(static)
  loop_box_3(valid, i, j, k)
        df[0](i,j,k) = (f(i,j,k) - f(i-1,j,k)) / dx[0];
  valid = rd.stagger(1);
#pragma omp parallel for default(shared) schedule(static)
  loop_box_3(valid, i, j, k)
        df[1](i,j,k) = (f(i,j,k) - f(i,j-1,k)) / dx[1];
  valid = rd.stagger(2);
#pragma omp parallel for default(shared) schedule(static)
  loop_box_3(valid, i, j, k)
        df[2](i,j,k) = (f(i,j,k) - f(i,j,k-1)) / dx[2];
}

template <>
void LevelOp<2>::computeCurl(const Tensor<Real, 2> *u, Tensor<Real, 2> *curlOfU) const
{
  auto dx = rd.spacing();
  Box<2> boxOfNodes(rd.lo(), rd.hi()-1);
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(boxOfNodes, i, j)
      curlOfU[0](i, j) = (u[1](i+1, j+1) - u[1](i, j+1)) / dx[0] - (u[0](i+1, j+1) - u[0](i+1, j)) / dx[1];
}

/*
 * The 3D curl goes here.
 */

/*
#define E0() i, j
#define E1(ed) i ed[0], j ed[1]
#define E2(ed, em) i ed[0] em[0], j ed[1] em[1]

template <>
template <int d>
Real LevelOp<2>::conv11(const Tensor<Real, 2> &F, const Vec<int, 2> &idx) const
{
  constexpr int ed[] = {d==0, d==1};
  constexpr int ed2[] = {ed[0]*2, ed[1]*2};
  constexpr int em[] = {d==1, d==0};
  const int i = idx[0], j = idx[1];
  Real a = ipow<2>(F(E1(-ed)) * (-1.0/16) + F(E0()) * (9.0/16) + F(E1(+ed)) * (9.0/16) + F(E1(+ed2)) * (-1.0/16));
  a += (1.0/12/16) * ipow<2>(F(E2(+ed, +em)) - F(E2(+ed, -em)) + F(E1(+em)) - F(E1(-em))); // transverse gradient
  return a;
}

template <>
template <int d, int m>
Real LevelOp<2>::conv12(const Tensor<Real, 2> &F, const Tensor<Real, 2> &G, const Vec<int, 2> &idx) const
{
  constexpr int ed[] = {d==0, d==1};
  constexpr int ed2[] = {ed[0]*2, ed[1]*2};
  constexpr int em[] = {m==0, m==1};
  constexpr int em2[] = {em[0]*2, em[1]*2};
  const int i = idx[0], j = idx[1];
  Real a = (F(E1(-em2)) * (-1.0/12) + F(E1(-em)) * (7.0/12) + F(E0()) * (7.0/12) + F(E1(+em)) * (-1.0/12))
      * (G(E1(-ed2)) * (-1.0/12) + G(E1(-ed)) * (7.0/12) + G(E0()) * (7.0/12) + G(E1(+ed)) * (-1.0/12));
  return a;
}

#undef E0
#undef E1
#undef E2*/
/*
template <>
void LevelOp<2>::computeConvection(const Tensor<Real, 2> *u, Tensor<Real, 2> *cnvu) const
{
  auto dx = rd.spacing();
  Box<2> gbx = rd.getGhostedBox();
  Tensor<Real, 2> ccTemp(gbx), nodeTemp(gbx);
  // d == 0
  Box<2> vbx = rd.inflate({1, 0});
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(vbx, i, j)
    ccTemp(i, j) = conv11<0>(u[0], {i, j});
  vbx = rd.stagger(0).inflate({-1, 0});
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(vbx, i, j)
    cnvu[0](i, j) = (ccTemp(i+1,j) * (-1.0/24) + ccTemp(i,j) * (27.0/24) + ccTemp(i-1,j) * (-27.0/24) + ccTemp(i-2,j) * (1.0/24)) / dx[0];
  vbx = Box<2>(rd.lo(), rd.hi()+1);
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(vbx, i, j)
    nodeTemp(i, j) = conv12<0,1>(u[0], u[1], {i, j});
  vbx = rd.stagger(0).inflate({-1, 0});
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(vbx, i, j)
    cnvu[0](i, j) += (nodeTemp(i,j+1) - nodeTemp(i,j)) / dx[1];
  // d == 1
  vbx = rd.inflate({0, 1});
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(vbx, i, j)
    ccTemp(i,j) = conv11<1>(u[1], {i, j});
  vbx = rd.stagger(1).inflate({0, -1});
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(vbx, i, j)
    cnvu[1](i,j) = (ccTemp(i,j+1) * (-1.0/24) + ccTemp(i,j) * (27.0/24) + ccTemp(i,j-1) * (-27.0/24) + ccTemp(i,j-2) * (1.0/24)) / dx[1];
  vbx = Box<2>(rd.lo(), rd.hi()+1);
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(vbx, i, j)
    nodeTemp(i,j) = conv12<1,0>(u[1], u[0], {i, j});
  vbx = rd.stagger(1).inflate({0, -1});
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(vbx, i, j)
    cnvu[1](i,j) += (nodeTemp(i+1,j) - nodeTemp(i,j)) / dx[0];
}*/

template <>
void LevelOp<2>::filterFace2Cell(const Tensor<Real, 2> *aFaceData, Tensor<Real, 3> &aCellData) const
{
  Box<2> valid = rd;
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(valid, i, j)
    aCellData(i, j, 0) = aFaceData[0](i,j) * (1.0/2) + aFaceData[0](i+1,j) * (1.0/2);
#pragma omp parallel for default(shared) schedule(static)
  loop_box_2(valid, i, j)
    aCellData(i, j, 1) = aFaceData[1](i,j) * (1.0/2) + aFaceData[1](i,j+1) * (1.0/2);
}

template <>
void LevelOp<3>::filterFace2Cell(const Tensor<Real, 3> *aFaceData, Tensor<Real, 4> &aCellData) const
{
  Box<3> valid = rd;
#pragma omp parallel for default(shared) schedule(static)
  loop_box_3(valid, i, j, k)
    aCellData(i, j, k, 0) = aFaceData[0](i,j,k) * (1.0/2) + aFaceData[0](i+1,j,k) * (1.0/2);
#pragma omp parallel for default(shared) schedule(static)
  loop_box_3(valid, i, j, k)
    aCellData(i, j, k, 1) = aFaceData[1](i,j,k) * (1.0/2) + aFaceData[1](i,j+1,k) * (1.0/2);
#pragma omp parallel for default(shared) schedule(static)
  loop_box_3(valid, i, j, k)
    aCellData(i, j, k, 2) = aFaceData[2](i,j,k) * (1.0/2) + aFaceData[2](i,j,k+1) * (1.0/2);
}

//============================================================
template class LevelOp<SpaceDim>;
