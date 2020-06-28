#include <fstream>
#include <sstream>
#include <iomanip>
#include "FiniteDiff/BOV.h"
#include "Core/TensorSlicing.h"

template <int Dim>
void writeBOV(const RectDomain<Dim> &aGrid,
              const Tensor<Real, Dim> &aData,
              const std::string &folder,
              const std::string &varname,
              int stamp,
              Real instant)
{
  assert(folder.empty() || folder[folder.size()-1] == '/');
  std::ostringstream prefix;
  prefix << varname << '.' << std::setfill('0') << std::setw(4) << stamp;
  dbgcout1 << "Writing " << prefix.str() << std::endl;
  Box<Dim> valid = aGrid;
  // first write the BOV Header
  do {
    std::ofstream header(folder + prefix.str() + ".bov");
    if(!header)
      break;
    Vec<Real,Dim> dx = aGrid.spacing();
    Vec<Real,Dim> locorner = dx * (aGrid.lo() + aGrid.getDelta() - 0.5);
    Vec<Real,Dim> bsize = dx * valid.size();
    header << "TIME: " << instant << "\n";
    header << "DATA_FILE: " << prefix.str() + ".dat\n";
    header << "DATA_SIZE: " << valid.size()[0] << " " << valid.size()[1] << " " << ((Dim>=3) ? (valid.size()[2]) : (1)) << "\n";
    header << "DATA_FORMAT: DOUBLE\n";
    header << "VARIABLE: " << varname << "\n";
    header << "DATA_ENDIAN: LITTLE\n";
    header << "CENTERING: zonal\n";
    header << "BRICK_ORIGIN: " << locorner[0] << " " << locorner[1] << " " << ((Dim>=3) ? (locorner[2]) : (0.0)) << "\n";
    header << "BRICK_SIZE: " << bsize[0] << ' ' << bsize[1] << " " << ((Dim>=3) ? (bsize[2]) : (0.0)) << "\n";
    header << "BYTE_OFFSET: " << sizeof(int) * (Dim+1) << "\n\n";
    header.close();
  } while(0);
  // then dump the data in binary format
  do {
    std::ofstream datafile(folder + prefix.str() + ".dat", std::ios::binary);
    if(!datafile)
      break;
    Tensor<Real,Dim> validData = aData.slice(valid);
    validData.dump(datafile);
    datafile.close();
  } while(0);
}

template void writeBOV(const RectDomain<SpaceDim> &aGrid,
                       const Tensor<Real, SpaceDim> &aData,
                       const std::string &folder,
                       const std::string &varname,
                       int stamp,
                       Real instant);

//============================================================

template <int Dim>
void writeBOVByComponent(const RectDomain<Dim> &aGrid,
                         const Tensor<Real, Dim> *aData,
                         int nComp,
                         const std::string &folder,
                         const std::string &varname,
                         int stamp,
                         Real instant)
{
  assert(folder.empty() || folder[folder.size()-1] == '/');
  std::ostringstream prefix;
  prefix << varname << '.' << std::setfill('0') << std::setw(4) << stamp;
  dbgcout1 << "Writing " << prefix.str() << std::endl;
  Box<Dim> valid = aGrid;
  Box<1> comps(0, nComp-1);
  // first write the BOV Header
  do {
    std::ofstream header(folder + prefix.str() + ".bov");
    if(!header)
      break;
    Vec<Real,Dim> dx = aGrid.spacing();
    Vec<Real,Dim> locorner = dx * (aGrid.lo() + aGrid.getDelta() - 0.5);
    Vec<Real,Dim> bsize = dx * valid.size();
    header << "TIME: " << instant << "\n";
    header << "DATA_FILE: " << prefix.str() + ".dat\n";
    header << "DATA_SIZE: " << valid.size()[0] << " " << valid.size()[1] << " " << ((Dim>=3) ? (valid.size()[2]) : (1)) << "\n";
    header << "DATA_FORMAT: DOUBLE\n";
    header << "VARIABLE: " << varname << "\n";
    header << "DATA_ENDIAN: LITTLE\n";
    header << "CENTERING: zonal\n";
    header << "BRICK_ORIGIN: " << locorner[0] << " " << locorner[1] << " " << ((Dim>=3) ? (locorner[2]) : (0.0)) << "\n";
    header << "BRICK_SIZE: " << bsize[0] << ' ' << bsize[1] << " " << ((Dim>=3) ? (bsize[2]) : (0.0)) << "\n";
    header << "BYTE_OFFSET: " << sizeof(int) * (Dim+2) << "\n";
    header << "DATA_COMPONENTS: " << nComp << "\n\n";
    header.close();
  } while(0);
  // then dump the data in binary format
  do {
    std::ofstream datafile(folder + prefix.str() + ".dat", std::ios::binary);
    if(!datafile)
      break;
    Tensor<Real, Dim+1> rearranged(enlarge(aGrid, comps, 0));
    ddfor(aGrid, [&](const Vec<int, Dim> &idx) {
      for(int n = 0; n < nComp; ++n)
        rearranged(enlarge(idx, n, 0)) = aData[n](idx);
    }, false);
    rearranged.dump(datafile);
    datafile.close();
  } while(0);
}

template void writeBOVByComponent(const RectDomain<SpaceDim> &aGrid,
                                  const Tensor<Real, SpaceDim> *aData,
                                  int nComp,
                                  const std::string &folder,
                                  const std::string &varname,
                                  int stamp,
                                  Real instant);
