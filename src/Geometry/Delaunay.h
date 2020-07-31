#ifndef DELAUNAY_H
#define DELAUNAY_H

#include "Core/Vec.h"

// The implementation is encapsulated in this class.
class DelaunayImpl;

/**
 * The main interface of Delaunay triangulation.
 */
class Delaunay
{
public:
  using rVec = Vec<Real, 2>;

  /**
   * The constructor.
   */
  Delaunay(Real aEps);

  /**
   * The destructor that frees the resources.
   */
  ~Delaunay();

  /**
   * Get the triangulation.
   * Typically this function is called twice :
   * During the first time the triangulation is performed and the number of triangles is returned.
   * During the second time, the user feeds in the memory for storing the triangles.
   * @param verts An array of nV*2 real numbers, storing the (x_i, y_i) pairs, 1<=i<=nV.
   * @param nV The number of vertices.
   * @param triOutput [out] An array of numTri*3 integers.
   * Every 3 integers specify the zero-based indices of the triangle.
   * Set to nullptr for the first call.
   * @param numTri [out] The number of triangles in the triangulation.
   */
  void triangulate(const rVec *verts, int nV, int *triOutput, int &numTri);

protected:
  DelaunayImpl *pImpl;
};

#endif //DELAUNAY_H
