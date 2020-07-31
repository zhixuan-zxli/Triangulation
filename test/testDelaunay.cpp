#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include "Geometry/Delaunay.h"

int main(int argc, char *argv[])
{
  using rVec = Vec<Real, 2>;
  const Real eps = 1e-12;
  const int nV = 1000;
  Real *vertices = new Real[nV*2];

  // Randomize the vertices
  std::default_random_engine::result_type seed = 0;
  std::default_random_engine re(seed);
  std::uniform_real_distribution<Real> uni(0.0, 1.0);
  for(int i = 0; i < 2*nV; ++i)
    vertices[i] = uni(re);

  // Output the vertices
  std::cout << "Vertices : " << std::endl;
  std::cout.precision(15);
  std::cout.setf(std::ios::scientific);
  for(int i = 0; i < nV; ++i)
    std::cout << std::setw(24) << vertices[2*i] << std::setw(24) << vertices[2*i+1] << "\n";

  // Perform the triangulation
  const rVec * pv = reinterpret_cast<const rVec *>(vertices);
  int numTri;
  Delaunay d(eps);
  d.triangulate(pv, nV, nullptr, numTri);

  // Obtain the triangulation
  int *tri = new int[numTri*3];
  d.triangulate(pv, nV, tri, numTri);

  // Output the triangulation
  std::cout << "\nTriangulation : " << std::endl;
  for(int i = 0; i < numTri; ++i)
    std::cout << std::setw(4) << tri[3*i] << std::setw(4) << tri[3*i+1] << std::setw(4) << tri[3*i+2] << "\n";

  std::cout << std::endl << std::endl;

  delete[] tri;
  delete[] vertices;
  return 0;
}
