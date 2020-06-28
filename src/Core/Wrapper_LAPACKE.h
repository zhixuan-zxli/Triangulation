#ifndef WRAPPER_LAPACKE_H
#define WRAPPER_LAPACKE_H

#ifndef USE_MKL
#include <cblas.h>
#include <lapacke.h>
#else
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#endif

#endif //WRAPPER_LAPACKE_H
