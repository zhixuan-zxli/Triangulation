#ifndef WRAPPER_OPENMP_H
#define WRAPPER_OPENMP_H

#ifndef OMP_PAR_LOWER_BOUND
#define OMP_PAR_LOWER_BOUND 1024
#endif // OMP_PAR_LOWER_BOUND

const int OMP_Par_Lower_Bound = OMP_PAR_LOWER_BOUND;

#ifdef USE_OPENMP

#include <omp.h>
#define enter_serial_if(cond) \
int __currentMaxThreads = omp_get_max_threads(); \
if((cond)) \
omp_set_num_threads(1);

#define exit_serial_if(cond) \
if((cond)) \
omp_set_num_threads(__currentMaxThreads);

#else

#define enter_serial_if(cond)
#define exit_serial_if(cond)

#endif // USE_OPENMP

#endif //WRAPPER_OPENMP_H
