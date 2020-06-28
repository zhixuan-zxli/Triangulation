#include <iostream>
#include "Core/Curve.h"
#include "Core/TensorSlicing.h"

int main()
{
  {
    Vec<Real, 2> a{1., 2.};
    Vec<Real, 2> b{-4., 3.};
    std::cout << a << b;
    std::cout << dot(a, b) << a + b << a + 2 << a * 3. << std::endl;
    Vec<Real, 1> c = -10.;
    std::cout << c + 1 << std::endl;
    std::cout << norm(b) << " " << dot(a, b) << std::endl;
    std::cout << reduce(Vec<Real, 3>{3., 2., 1.}) << enlarge(b, -1.) << std::endl;
  }
  //================================================
  {
    RealPolynomial<3> p{0.0, -1.0, 1.0};
    RealPolynomial<3> q{-1.0, 2.0, 0.5};
    auto k = p * q;
    std::cout << root(p, 1.1, 1e-10, 10) << std::endl;
    auto m = p.der();
    auto n = p.translate(0.5);
  }
  //================================================
  {
    using Crv = Curve<2, 2>;
    auto rect = createRect<2>({0.0, 0.0}, {2.0, 1.0});
    std::vector<Crv> pieces;
    rect.split({1.0, 3.0, 4.0, 5.0}, pieces, 1e-12);
    std::cout << area(rect) << ", " << arclength(rect) << std::endl;

    Crv c;
    for (const auto &a : pieces)
      c.concat(a);
    std::cout << c(2.5) << std::endl;
  }
  //================================================
  {
    Tensor<Real, 1> a(5);
    a = {-1, 0, 1, 2, 3};
    std::cout << a << std::endl;

    Tensor<Real,2> b(Vec<int,2> {5, 5});
    b = 0.0;
    b(1,0) = 2.0;
    b(0,1) = 3.0;
    b(2) = -1.0;
    std::cout << b << std::endl;

    Tensor<Real,2> c(b);
    c(0, 2) = -2.0;
    std::cout << b << std::endl;

    Tensor<Real,2> d = c.slice(Box<2>(3, 4));
    std::cout << d << std::endl;
    c.slice(1, 4) = -c.slice(1, 0);
    c.slice(0, 4) = c.slice(0, 0) * 0.5 + a;
    std::cout << c << std::endl;

    std::cout << sum(b) << std::endl;
    std::cout << norm(b, 2) << std::endl;
    std::cout << norm(b.slice(Box<2>(0, 2)), 1) << std::endl;
  }
  {
    Tensor<Real,2> a(3);
    a = {-1, 0, 1, 2, 3, 4, 5, 6, 7};
    Tensor<Real,2> b(Box<2>(1, 3));
    b = 1.0;
    Tensor<Real,2> c = a+b*2;
    std::cout << c << std::endl;
  }

  return 0;
}
