add_executable(testCore testCore.cpp)
target_link_libraries(testCore ${LIB_NAME})
add_executable(testFiniteDiff testFiniteDiff.cpp)
target_link_libraries(testFiniteDiff ${LIB_NAME})
add_executable(testMG4Poisson testMG4Poisson.cpp)
target_link_libraries(testMG4Poisson ${LIB_NAME})
add_executable(testDelaunay testDelaunay.cpp)
target_link_libraries(testDelaunay ${LIB_NAME})

enable_language(Fortran)
add_executable(testPerf testPerf.cpp relax.F)
target_link_libraries(testPerf ${LIB_NAME})
