#ifndef TESTUTILITY_H
#define TESTUTILITY_H

#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>

template <class T>
inline
std::string oneOver(const T &denom) {
  std::ostringstream oss;
  oss << "1/" << denom;
  return oss.str();
}

template <std::size_t numOfNorms>
inline
void printConvergenceTable(const int *gridSize,
                           const std::vector<std::array<Real, numOfNorms>> &errnorm)
{
  const int numCompHier = errnorm.size();
  const int w = 10;
  const char *ntHeader[] = {"$L^\\infty$", "$L^1$", "$L^2$"};
  std::cout << "\n\n" << std::setw(w) << "h";
  for(int n = numCompHier-1; n >= 0; --n)
    std::cout << " & " << std::setw(w) << oneOver(gridSize[n]) << " & " << std::setw(w) << "rate";
  std::cout << " \\\\" << std::endl;
  std::cout.precision(2);
  for(std::size_t p=0; p<numOfNorms; ++p) {
    std::cout << std::setw(w) << ntHeader[p%3];
    for(int n = numCompHier-1; n >= 0; --n) {
      std::cout << " & " << std::scientific << std::setw(w) << errnorm[n][p];
      if(n != 0)
        std::cout << " & " << std::fixed << std::setw(w) << log(errnorm[n-1][p] / errnorm[n][p]) / log(1.0 * gridSize[n] / gridSize[n-1]);
    }
    std::cout << " \\\\" << std::endl;
  }
}

inline std::string getTestIdentifier(const std::string &nameOfTest)
{
  std::ostringstream oss;
  std::time_t t = std::time(nullptr);
  oss << nameOfTest << std::put_time(std::localtime(&t), "-%y%m%d-%H%M");
  return oss.str();
}

struct CPUTimer
{
  using HRC = std::chrono::high_resolution_clock;
  std::chrono::time_point<HRC>  start;
  CPUTimer() { reset(); }
  void reset() { start = HRC::now(); }
  double operator() () const {
    std::chrono::duration<double> e = HRC::now() - start;
    return e.count();
  }
};

#endif //TESTUTILITY_H
