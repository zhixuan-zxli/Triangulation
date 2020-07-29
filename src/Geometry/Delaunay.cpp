#include <utility>
#include "Geometry/Delaunay.h"

/**
 * The implementation class for Delaunay triangulation.
 */
class DelaunayImpl
{
  // gemetric primitive
protected:
  enum { Dim = 2 };
  using rVec = Vec<Real, 2>;

  /**
   * Return > 0 if abc is in counter-clockwise order.
   */
  static Real ccw(const rVec &a, const rVec &b, const rVec &c)
  {
    return (b[0]*c[1] + c[0]*a[1] + a[0]*b[1])
        - (c[0]*b[1] + a[0]*c[1] + b[0]*a[1]);
  }

  /**
   * Return > 0 if d is inside the circumcircle of abc.
   */
  static Real inCircle(const rVec &a, const rVec &b, const rVec &c, const rVec &d)
  {
    Real a2 = dot(a, a);
    Real b2 = dot(b, b);
    Real c2 = dot(c, c);
    Real d2 = dot(d, d);
    Real det1 = (b[0]*c[1]*d2 + b[1]*c2*d[0] + b2*c[0]*d[1]) - (b[0]*c2*d[1] + b[1]*c[0]*d2 + b2*c[1]*d[0]);
    Real det2 = (a[0]*c[1]*d2 + a[1]*c2*d[0] + a2*c[0]*d[1]) - (a[0]*c2*d[1] + a[1]*c[0]*d2 + a2*c[1]*d[0]);
    Real det3 = (a[0]*b[1]*d2 + a[1]*b2*d[0] + a2*b[0]*d[1]) - (a[0]*b2*d[1] + a[1]*b[0]*d2 + a2*b[1]*d[0]);
    Real det4 = (a[0]*b[1]*c2 + a[1]*b2*c[0] + a2*b[0]*c[1]) - (a[0]*b2*c[1] + a[1]*b[0]*c2 + a2*b[1]*c[0]);
    return -det1 + det2 - det3 + det4;
  }

  // data structure
protected:

  using EdgeRef = unsigned long; // encode with a pointer to a triangle an an edge index in the least-significant 2 bits

  struct Triangle {
    const rVec * vertices[3]; // The pointers to the 3 vertices
    EdgeRef      incident[3]; // The pointers to the incident faces and edges.
    // [0] points to the triangle and its edge incident to the edge v[0]-v[1], etc.
    // unsigned long userData;
  };

  static EdgeRef pack(Triangle *pT, unsigned char side)
  {
    return reinterpret_cast<EdgeRef>(pT) | side;
  }

  static void unpack(EdgeRef u, Triangle * &pT, unsigned char &side)
  {
    pT = reinterpret_cast<Triangle*>(u & (~0x03));
    side = u & 0x03;
  }

  static unsigned char oppoVertex(unsigned char side) { return (side+2)%3; }

  /**
   * Bridge the origin of eLeft to the origin of eRight.
   * eLeft and eRight may be null.
   * @return The new edges.
   */
  static std::pair<EdgeRef, EdgeRef> bridge(EdgeRef eLeft, EdgeRef eRight)
  {
    Triangle *tLeft, *tRight;
    unsigned char sLeft, sRight;
    unpack(eLeft, tLeft, sLeft);
    unpack(eRight, tRight, sRight);
    assert(eLeft == 0 || tLeft->vertices[oppoVertex(sLeft)] == nullptr);
    assert(eRight == 0 || tRight->vertices[oppoVertex(sRight)] == nullptr);
    Triangle *tDown = new Triangle;
    tDown->vertices[2] = nullptr;
    Triangle *tUp = new Triangle;
    tUp->vertices[2] = nullptr;
    tDown->incident[0] = pack(tUp, 0);
    tUp->incident[0] = pack(tDown, 0);
    // bridge the left part
    if(eLeft == 0) {
      tDown->incident[1] = pack(tUp, 2);
      tUp->incident[2] = pack(tDown, 1);
    } else {
      Triangle *tLeftUp;
      unsigned char sLeftUp;
      unpack(tLeft->incident[(sLeft+2)%3], tLeftUp, sLeftUp);
      tLeft->incident[(sLeft+2)%3] = pack(tDown, 1);
      tDown->incident[1] = pack(tLeft, (sLeft+2)%3);
      tLeftUp->incident[sLeftUp] = pack(tUp, 2);
      tUp->incident[2] = pack(tLeftUp, sLeftUp);
      tDown->vertices[1] = tUp->vertices[0] = tLeft->vertices[sLeft];
    }
    // bridge the right part
    if(eRight == 0) {
      tDown->incident[2] = pack(tUp, 1);
      tUp->incident[1] = pack(tDown, 2);
    } else {
      Triangle *tRightDown;
      unsigned char sRightDown;
      unpack(tRight->incident[(sRight+2)%3], tRightDown, sRightDown);
      tRight->incident[(sRight+2)%3] = pack(tUp, 1);
      tUp->incident[1] = pack(tRight, (sRight+2)%3);
      tRightDown->incident[sRightDown] = pack(tDown, 2);
      tDown->incident[2] = pack(tRightDown, sRightDown);
      tDown->vertices[0] = tUp->vertices[1] = tRight->vertices[sRight];
    }
    return std::make_pair(pack(tDown, 0), pack(tUp, 0));
  }

  /**
   * Flip the edge e as the diagonal of a quadrilateral.
   */
  static std::pair<EdgeRef, EdgeRef> flip(EdgeRef e)
  {
    assert(e != 0);
    Triangle *eLeft, *eRight, *eLeftUp, *eLeftDown, *eRightUp, *eRightDown;
    unsigned char *sLeft, sRight, sLeftUp, sLeftDown, sRightUp, sRightDown;
    unpack(e,                              eLeft,      sLeft);
    unpack(eLeft->incident[(sLeft+1)%3],   eLeftUp,    sLeftUp);
    unpack(eLeft->incident[(sLeft+2)%3],   eLeftDown,  sLeftDown);
    unpack(eLeft->incident[sLeft],         eRight,     sRight);
    unpack(eRight->incident[(sRight+1)%3], eRightDown, sRightDown);
    unpack(eRight->incident[(sRight+2)%3], eRightUp,   sRightUp);
    // Reshape the triangles. Now eLeft becomes eDown, eRight becomes eUp.
    const rVec *v[4] = { eLeft->vertices[sLeft],
                         eLeft->vertices[(sLeft+1)%3],
                         eLeft->vertices[(sLeft+2)%3],
                         eRight->vertices[(sRight+2)%3] };
    eLeft->vertices[0] = v[3];
    eLeft->vertices[1] = v[2];
    eLeft->vertices[2] = v[0];
    eRight->vertices[0] = v[2];
    eRight->vertices[1] = v[3];
    eRight->vertices[2] = v[1];
    // Re-establish the connectivity.
    eLeft->incident[0]  = pack(eRight,     0);
    eLeft->incident[1]  = pack(eLeftDown,  sLeftDown);
    eLeft->incident[2]  = pack(eRightDown, sRightDown);
    eRight->incident[0] = pack(eLeft,      0);
    eRight->incident[1] = pack(eRightUp,   sRightUp);
    eRight->incident[2] = pack(eLeftUp,    sLeftUp);
    eLeftUp->incident[sLeftUp]       = pack(eRight, 2);
    eLeftDown->incident[sLeftDown]   = pack(eLeft,  1);
    eRightUp->incident[sRightUp]     = pack(eRight, 1);
    eRightDown->incident[sRightDown] = pack(eLeft,  2);
    // Return the new edges.
    return std::make_pair(pack(eLeft, 0), pack(eRight, 0));
  }

  // computational routines
protected:
  /**
   *
   * @return One of the triangle in the triangulation.
   */
  Triangle *Guibas_Stolfi_DT(const rVec *pVertices, std::size_t numVertices);

  /**
   * The recursive version of Guibas-Stolfi triangulation.
   * @pre The vertices are sorted in x-increasing order.
   * @param triOnHull [0] for the cw edge from the leftmost vertex,
   *                  [1] for the cw edge from the rightmost vertex,
   *                  [2] for the ccw edge from the bottommost vertex,
   *                  [3] for the cw edge from the topmost vertex.
   */
  void Guibas_Stolfi_recur(const rVec *v,
                           std::size_t nV,
                           EdgeRef *triOnHull)
  {
    assert(nV >= 2);
    // Case 1. Two vertices.
    if(nV == 2) {
      auto ep = bridge(0, 0);
      Triangle *tDown, *tUp;
      unsigned char sDown, sUp;
      unpack(ep.first, tDown, sDown);
      unpack(ep.second, tUp, sUp);
      tDown->vertices[0] = tUp->vertices[1] = &v[1];
      tDown->vertices[1] = tUp->vertices[0] = &v[0];
      triOnHull[0] = pack(tUp, 0);
      triOnHull[1] = pack(tDown, 0);
    // Case 2. Three vertices.
    } else if(nV == 3) {
      auto ep01 = bridge(0, 0);
      Triangle *tDown, *tUp, *tDown_, *tUp_;
      unsigned char sDown, sUp, sDown_, sUp_;
      unpack(ep01.first, tDown, sDown);
      unpack(ep01.second, tUp, sUp);
      tDown->vertices[0] = tUp->vertices[1] = &v[1];
      tDown->vertices[1] = tUp->vertices[0] = &v[0];
      auto ep12 = bridge(pack(tDown, 0), 0);
      unpack(ep12.first, tDown_, sDown_);
      unpack(ep12.second, tUp_, sUp_);
      tDown_->vertices[0] = tUp_->vertices[1] = &v[2];
      if(ccw(v[0], v[1], v[2]) > epsilon) {
        auto ep20 = flip(ep01.second, ep12.second);
        triOnHull[0] = ep20.second;
        triOnHull[1] = ep12.first;
      } else if(ccw(v[0], v[1], v[2]) < -epsilon) {
        auto ep20 = flip(ep12.first, ep01.first);
        triOnHull[0] = ep01.second;
        triOnHull[1] = ep20.second;
      } else {
        // v0, v1, v2 almost colinear
        triOnHull[0] = ep01.second;
        triOnHull[1] = ep12.first;
      }
    // Case 3. Four or more vertices
    } else {
    }
  }

  // The interface
public:
  /**
   * The default constructor accepting the tolerance.
   * @param Epsilon
   */
  DelaunayImpl(Real aEpsilon) : epsilon(aEpsilon)
  {
  }

  Real epsilon;
};
