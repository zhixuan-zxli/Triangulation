#ifndef TENSORSLICE_H
#define TENSORSLICE_H

#include "Core/TensorExpr.h"
#include "Core/VecCompare.h"

// always provide a zero-base indexing to the data
template <class T, int Dim>
class TensorSlice : public TensorExpr<const TensorSlice<T, Dim> &>
{
public:
  using iVec = Vec<int, Dim>;

  TensorSlice() = delete;
  // copy constructor of TensorSlice. do not mean copying the underlying data
  TensorSlice(const TensorSlice<T, Dim> &) = default;

  TensorSlice<T, Dim>& operator=(const T &a);

  // assignment of the underlying data, not a copy assignment
  TensorSlice<T, Dim> &operator=(const TensorSlice<T, Dim> &rhs);

  // assign from expressions
  template <class TExpr>
  TensorSlice<T, Dim> &operator=(const TensorExpr<TExpr> &expr);

  // support for template expressions
public:
  const T& at(int i) const { return zData[i*stride[0]]; }
  T&       at(int i)       { return zData[i*stride[0]]; }
  const T& at(int i, int j) const { return zData[i*stride[0] + j*stride[1]]; }
  T&       at(int i, int j)       { return zData[i*stride[0] + j*stride[1]]; }
  const T& at(int i, int j, int k) const { return zData[i*stride[0] + j*stride[1] + k*stride[2]]; }
  T&       at(int i, int j, int k)       { return zData[i*stride[0] + j*stride[1] + k*stride[2]]; }
  const T& at(int i, int j, int k, int l) const { return zData[i*stride[0] + j*stride[1] + k*stride[2] + l*stride[3]]; }
  T&       at(int i, int j, int k, int l)       { return zData[i*stride[0] + j*stride[1] + k*stride[2] + l*stride[3]]; }

  Box<Dim> box() const { return Box<Dim>(0, sz-1); }

protected:
  template <class S, int E> friend class Tensor;

  TensorSlice(T *_z, const iVec &_sz, const iVec &_st) : zData(_z), sz(_sz), stride(_st)
  {
  }

  T * __restrict__ zData;
  iVec sz;
  iVec stride;
};

//================================================

template <class T, int Dim>
TensorSlice<T, Dim>& TensorSlice<T, Dim>::operator=(const T &a)
{
  if(Dim==1) {
#pragma omp parallel for default(shared) schedule (static)
    loop_sz_1(sz, i)
      at(i) = a;
  } else if(Dim==2) {
#pragma omp parallel for default(shared) schedule (static)
    loop_sz_2(sz, i, j)
      at(i, j) = a;
  } else if(Dim == 3) {
#pragma omp parallel for default(shared) schedule (static)
    loop_sz_3(sz, i, j, k)
      at(i, j, k) = a;
  } else if(Dim == 4) {
#pragma omp parallel for default(shared) schedule (static)
    loop_sz_4(sz, i, j, k, l)
      at(i, j, k, l) = a;
  }
  return *this;
}

template <class T, int Dim>
inline
TensorSlice<T, Dim>& TensorSlice<T, Dim>::operator=(const TensorSlice<T, Dim> &rhs)
{
  *this = static_cast<const TensorExpr<const TensorSlice<T, Dim> &> &>(rhs);
  return *this;
}

template <class T, int Dim>
template <class TExpr>
inline
TensorSlice<T, Dim>& TensorSlice<T, Dim>::operator=(const TensorExpr<TExpr> &expr)
{
  assert((VecCompare<int,Dim>().compare(sz, expr.box().size())) == 0);
  if(Dim==1) {
#pragma omp parallel for default(shared) schedule (static)
    loop_sz_1(sz, i)
      at(i) = expr.at(i);
  } else if(Dim==2) {
#pragma omp parallel for default(shared) schedule (static)
    loop_sz_2(sz, i, j)
      at(i, j) = expr.at(i, j);
  } else if(Dim == 3) {
#pragma omp parallel for default(shared) schedule (static)
    loop_sz_3(sz, i, j, k)
      at(i, j, k) = expr.at(i, j, k);
  } else if(Dim == 4) {
#pragma omp parallel for default(shared) schedule (static)
    loop_sz_4(sz, i, j, k, l)
      at(i, j, k, l) = expr.at(i, j, k, l);
  }
  return *this;
}

//================================================
// to create tensor slicing

template <class T, int Dim>
inline
TensorSlice<T, Dim> Tensor<T, Dim>::slice(const Box<Dim> &pbox) const
{
  return TensorSlice<T, Dim>(const_cast<T*>(aData + dot(pbox.lo() - bx.lo(), stride)), pbox.size(), stride);
}

template <class T, int Dim>
inline
TensorSlice<T, Dim-1> Tensor<T, Dim>::slice(int D, int a) const
{
  auto newlo = bx.lo();
  newlo[D] = a;
  return TensorSlice<T, Dim-1>(const_cast<T*>(aData + dot(newlo - bx.lo(), stride)), reduce(bx.size(), D), reduce(stride, D));
}


#endif //TENSORSLICE_H
