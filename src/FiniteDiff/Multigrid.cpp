#include <iomanip>
#include "Core/TensorExpr.h"
#include "Core/Wrapper_OpenMP.h"
#include "FiniteDiff/Multigrid.h"

template <int Dim>
Real Multigrid<Dim>::solve(Tensor<Real, Dim> &phi, const Tensor<Real, Dim> &rhs, const Tensor<Real, Dim - 1> *cData) const
{
  Real cur_rsd[2];
  Real init_rsd, rel_rsd, redc;
  const RectDomain<Dim> &rd = opHier[0]->getDomain();
  Box<Dim> gbx = rd.getGhostedBox();
  Tensor<Real,Dim> rsd(gbx);
  Tensor<Real,Dim> corr(gbx);

  int k;
  for(k = 0; ; ++k) {
    opHier[0]->fillGhosts(phi, rhs, cData);
    opHier[0]->computeResidual(phi, rsd, rhs);
    cur_rsd[1] = opHier[0]->computeNorm(rsd);
    if(k == 0) {
      init_rsd = cur_rsd[0] = cur_rsd[1];
    } else {
      rel_rsd = cur_rsd[1] / init_rsd;
      redc = cur_rsd[0] / cur_rsd[1];
      cur_rsd[0] = cur_rsd[1];
      dbgcout1 << "Multigrid iter " << k
               << ", rel. rsd. = " << std::scientific << std::setprecision(3) << rel_rsd
               << ", redc = " << std::defaultfloat << std::setprecision(3) << redc
               << "\n";
      if (redc < stallThr || rel_rsd < relTol || k >= maxIter)
        break;
    }
    corr = 0.0;
    VCycle(0, corr, rsd, 0);
    phi = phi + corr;
  }
  dbgcout << "Multigrid exits after " << k << " iters, rel. rsd. = " << rel_rsd << std::endl;
  return rel_rsd;
}

template <int Dim>
template <class T>
void Multigrid<Dim>::VCycle(int depth, Tensor<Real, Dim> &phi, const Tensor<Real, Dim> &rhs, const T &cData) const
{
  if(depth == (int)(opHier.size() - 1)) {
    opHier[depth]->solveBottom(phi, rhs, numRelaxes[2]);
    return;
  }

  const RectDomain<Dim> &aDomain = opHier[depth]->getDomain();
  Box<Dim> gbx = aDomain.getGhostedBox();
  Tensor<Real,Dim> temp(gbx);
  // pre-smooth
  auto smoother = [&](Tensor<Real,Dim> &aPhi, Tensor<Real,Dim> &aTemp) {
    opHier[depth]->fillGhosts(aPhi, rhs, cData);
    opHier[depth]->applySmoother(aPhi, aTemp, rhs);
    opHier[depth]->fillGhosts(aTemp, rhs, cData);
    opHier[depth]->applySmoother(aTemp, aPhi, rhs);
  };
  enter_serial_if(aDomain.volume() <= OMP_Par_Lower_Bound)
  for(int k=0; k<numRelaxes[0]; ++k)
    smoother(phi, temp);

  Tensor<Real,Dim> fineRsd(gbx);
  opHier[depth]->fillGhosts(phi, rhs, cData);
  opHier[depth]->computeResidual(phi, fineRsd, rhs);
  // transfer the residual
  RectDomain<Dim> coarseGrid = opHier[depth+1]->getDomain();
  Box<Dim> cgbx = coarseGrid.getGhostedBox();
  Tensor<Real, Dim> coarseRsd(cgbx);
  trHier[depth]->applyRestriction(fineRsd, coarseRsd);
  // recursive call
  Tensor<Real, Dim> cphi(cgbx);
  cphi = 0.0;
  VCycle(depth+1, cphi, coarseRsd, 0);
  // prolongation and correction
  trHier[depth]->incrementalProlong(cphi, phi);

  // post-relax
  for(int k=0; k<numRelaxes[1]; ++k)
    smoother(phi, temp);
  opHier[depth]->fillGhosts(phi, rhs, cData);
  exit_serial_if(aDomain.volume() <= OMP_Par_Lower_Bound)
}

//============================================================
template class MGLevelOp<SpaceDim>;
template class Multigrid<SpaceDim>;
