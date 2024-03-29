add_library(${LIB_NAME}
        Core/Config.h
        Core/Config.cpp
        Core/Vec.h
        Core/VecCompare.h
        Core/Box.h
        Core/Interval.h
        Core/numlib.h
        Core/numlib.cpp
        Core/static_for.h
        Core/Polynomial.h
        Core/Curve.h
        Core/Curve.cpp
        Core/Tensor.h
        Core/TensorExpr.h
        Core/TensorSlice.h
        Core/RowSparse.h
        Core/Wrapper_LAPACKE.h
        Core/Wrapper_OpenMP.h
        FiniteDiff/RectDomain.h
        FiniteDiff/FuncFiller.h
        FiniteDiff/Intergrid.h
        FiniteDiff/Intergrid.cpp
        FiniteDiff/GhostFiller.h
        FiniteDiff/GhostFiller.cpp
        FiniteDiff/LevelOp.h
        FiniteDiff/LevelOp.cpp
        FiniteDiff/Multigrid.h
        FiniteDiff/Multigrid.cpp
        Geometry/Delaunay.cpp
        Geometry/Delaunay.h
        )
if(USE_MKL)
    target_link_libraries(${LIB_NAME} -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl)
else()
    target_link_libraries(${LIB_NAME} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES} gfortran)
endif()
