#ifndef ERK_H
#define ERK_H

struct RK4
{
  static const int nStages = 4;
  static constexpr Real c[] = {
      0.0, 0.5, 0.5, 1.0
  };
  static constexpr Real b[] = {
      1.0/6, 1.0/3, 1.0/3, 1.0/6
  };
  static constexpr Real a[][nStages] = {
      {0, 0, 0, 0},
      {0.5, 0, 0, 0},
      {0, 0.5, 0, 0},
      {0, 0, 1, 0}
  };
};

constexpr Real RK4::c[];
constexpr Real RK4::b[];
constexpr Real RK4::a[][RK4::nStages];

#endif //ERK_H
