#include "black_nufft.h"
#include "fftw3.h"
#include "fftw3-mpi.h"
#include <deal.II/base/exceptions.h>

using namespace tbb;

// Function to compute the next integer divisible for 2, 3, and 5.
int next235(double in)
{
  int result;
  int numdiv = 0;
  int next235 = 2 * int(in/2.+.9999);
  if (next235 <=0)
    next235 = 2;
FOO:
  numdiv = next235;
  while (numdiv/2*2 == numdiv)
    {
      numdiv = numdiv /2;
    }
  while (numdiv/3*3 == numdiv)
    {
      numdiv = numdiv /3;
    }
  while (numdiv/5*5 == numdiv)
    {
      numdiv = numdiv /5;
    }
  if (numdiv == 1)
    return result = next235;
  next235 = next235+2;
  goto FOO;
}

BlackNUFFT::BlackNUFFT(const std::vector<std::vector<double> > &in_grid, const std::vector<double> &in, std::vector<std::vector<double> > &out_grid, std::vector<double> &out, MPI_Comm comm_in)
  :
  input_grid(in_grid),
  output_grid(out_grid),
  input_vector(in),
  output_vector(out),
  mpi_communicator(comm_in),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout(std::cout,
        (this_mpi_process
         == 0)),
  computing_timer(mpi_communicator,
                  pcout,
                  TimerOutput::summary,
                  TimerOutput::wall_times)
{}

// A simple initialiser that resizes the grid parameters and the number of points.
// We have put an Assert to check that the requested tolerance is right.

void BlackNUFFT::init_nufft(double eps, bool fft_bool, std::string gridding_input, std::string fft_input)
{
  TimerOutput::Scope t(computing_timer, " Initialisation ");
  nj = input_grid[0].size();
  nk = output_grid[0].size();
  xb.resize(3);
  sb.resize(3);
  xm.resize(3);
  sm.resize(3);
  Assert((eps >= 1e-33) && (eps <= 1e-1), ExcNotImplemented());
  epsilon = eps;
  fft_backward = fft_bool;
  gridding = gridding_input;
  fft_type = fft_input;
  pcout<<"Using "<<gridding<<" as gridding tool and "<<fft_type<<" as backend FFT library"<<std::endl;
}


// Inside this function we need to create the 4 index sets we need in the computations.

void BlackNUFFT::create_index_sets()
{
  TimerOutput::Scope t(computing_timer, " Computing IndexSets ");


  if (fft_type == "FFTW")
    {

      ptrdiff_t tmp_local_nf3, tmp_local_nf3_start;
      fftw_mpi_local_size_3d(nf3, nf2, nf1, mpi_communicator, &tmp_local_nf3, &tmp_local_nf3_start);
      local_nf3 = (types::global_dof_index) tmp_local_nf3;
      local_nf3_start = (types::global_dof_index) tmp_local_nf3_start;
      Assert(local_nf3*nf1*nf2*2 <= std::numeric_limits<unsigned int>::max(), ExcMessage("The number of local elements must be less than 2^32, please increase the number of MPI processors."));
      // fine_grid_data.reinit(complete_index_set(2*(nf1*nf2*nf3)),mpi_communicator);

      // We create this index set following the repartition of fftw3.
      // We create this index set following the repartition of fftw3.
      fftw3_set.set_size(nf1*nf2*nf3*2);
      fftw3_set.add_range(nf1*nf2*(local_nf3_start)*2, nf1*nf2*(local_nf3_start+local_nf3)*2);

      // We create this index set following the repartition of fftw3. We need to be sure that all the things that influence
      // the output are included here. This will be the relevant index set for the distributed array.
      fftw3_output_set.set_size(2*nf3*nf2*nf1);
      types::global_dof_index ghost1, ghost2;
      if (local_nf3_start>nspread)
        ghost1=nspread;
      else
        ghost1=0;//local_nf3_start;
      if (nf3-local_nf3_start-local_nf3>nspread)
        ghost2=nspread;
      else
        ghost2=0;//nf3-local_nf3_start-local_nf3;
      fftw3_output_set.add_range(nf1*nf2*(local_nf3_start-ghost1)*2, nf1*nf2*(local_nf3_start)*2);
      fftw3_output_set.add_range(nf1*nf2*(local_nf3_start+local_nf3)*2, nf1*nf2*(local_nf3_start+local_nf3+ghost2)*2);

      fine_grid_data.reinit(fftw3_set, fftw3_output_set, mpi_communicator);

      // We create the input set associated with the set needed by fftw 3d.
      input_set.set_size(nj);
      for (types::global_dof_index j=0; j<nj; ++j)
        {
          auto jb1 = types::global_dof_index(double(nf1/2) + (input_grid[0][j]-xb[0])/hx);
          auto jb2 = types::global_dof_index(double(nf2/2) + (input_grid[1][j]-xb[1])/hy);
          auto jb3 = types::global_dof_index(double(nf3/2) + (input_grid[2][j]-xb[2])/hz);
          if (fftw3_set.is_element(2 * (jb1 + jb2*nf1 + jb3*nf1*nf2)))
            {
              input_set.add_index(j);
            }

        }
      input_set.compress();
    }
  else
    {
      AssertThrow(true, ExcNotImplemented());
    }
  // We create the additional input sets needed by the accelerating version of the gridding.
  create_index_sets_for_first_gridding();

  output_set.set_size(nk);
  for (types::global_dof_index k=0; k<nk; ++k)
    {
      auto kb1 = types::global_dof_index(double(nf1/2) + (output_grid[0][k]-sb[0])/hs);
      auto kb2 = types::global_dof_index(double(nf2/2) + (output_grid[1][k]-sb[1])/ht);
      auto kb3 = types::global_dof_index(double(nf3/2) + (output_grid[2][k]-sb[2])/hu);

      if (fftw3_set.is_element((kb1+kb2*nf1+kb3*nf1*nf2)*2))
        {
          output_set.add_index(k);
        }
    }
  output_set.compress();
  pcout<<" Input work Balance : "<<input_set.n_elements()<<" elements over "<<input_set.size()<<std::endl;
  pcout<<" Output work Balance : "<<output_set.n_elements()<<" elements over "<<output_set.size()<<std::endl;
}

// The following function creates the index sets stored in grid_sets to speed up
// the first gridding. The idea is to split the "owned" input points in two well split groups
// that can run completely in parallel. This should speed up things. Since MPI splits the grid
// along the third dimension we have chosen to split the second one. We use 2*nspread to identify
// regions that can't have racing condition and we subdivide the grid in odd and even part. We will
// perform two TBB cycle on odd and even separately.
void BlackNUFFT::create_index_sets_for_first_gridding(const unsigned int sets_number)
{
  grid_sets.clear();
  grid_sets.resize(sets_number);
  std::vector<std::vector<IndexSet> > helper(sets_number);

  types::global_dof_index dividend = nf2 / (2*nspread);
  types::global_dof_index rest = nf2 % (2*nspread);
  for (unsigned int i = 0; i<sets_number; ++i)
    {
      grid_sets[i].clear();
      grid_sets[i].resize((dividend+1)/2);
      helper[i].resize((dividend+1)/2);
      for (types::global_dof_index j = 0; j<helper[i].size(); ++j)
        {
          grid_sets[i][j].clear();
          helper[i][j].clear();
          grid_sets[i][j].set_size(input_set.size());
          helper[i][j].set_size(fftw3_set.size());
        }

    }
  // We use helper to split the grid. helper[0] holds the odd parts and helper[1] the even ones.
  for (types::global_dof_index i = 0; i<dividend; i=i+2)
    {
      for (types::global_dof_index k3 = local_nf3_start; k3 < (local_nf3_start + local_nf3); ++k3)
        {
          helper[0][i/2].add_range(2*(k3*nf1*nf2 + i * 2 * nspread * nf1), 2*(k3*nf1*nf2 + (i+1) * 2 * nspread * nf1));
        }
    }
  for (types::global_dof_index i = 1; i<dividend; i=i+2)
    {
      for (types::global_dof_index k3 = local_nf3_start; k3 < (local_nf3_start + local_nf3); ++k3)
        {
          helper[1][i/2].add_range(2*(k3*nf1*nf2 + i * 2 * nspread * nf1), 2*(k3*nf1*nf2 + (i+1) * 2 * nspread * nf1));
        }
    }
  if (dividend % 2 == 0)
    for (types::global_dof_index k3 = local_nf3_start; k3 < (local_nf3_start + local_nf3); ++k3)
      {
        helper[1].back().add_range(2*(k3*nf1*nf2 + dividend * 2 * nspread * nf1), 2*(k3*nf1*nf2 + (dividend * 2 * nspread + rest) * nf1));
      }
  else
    for (types::global_dof_index k3 = local_nf3_start; k3 < (local_nf3_start + local_nf3); ++k3)
      {
        helper[0].back().add_range(2*(k3*nf1*nf2 + dividend * 2 * nspread * nf1), 2*(k3*nf1*nf2 + (dividend * 2 * nspread + rest) * nf1));
      }



  for (types::global_dof_index i=0; i<helper.size(); ++i)
    for (types::global_dof_index j=0; j<helper.size(); ++j)
      {
        helper[i][j].compress();
      }
  for (auto j : input_set)
    {
      auto jb1 = types::global_dof_index(double(nf1/2) + (input_grid[0][j]-xb[0])/hx);
      auto jb2 = types::global_dof_index(double(nf2/2) + (input_grid[1][j]-xb[1])/hy);
      auto jb3 = types::global_dof_index(double(nf3/2) + (input_grid[2][j]-xb[2])/hz);
      for (types::global_dof_index jj=0; jj<helper[0].size(); ++jj)
        if (helper[0][jj].is_element(2 * (jb1 + jb2*nf1 + jb3*nf1*nf2)))
          {
            grid_sets[0][jj].add_index(j);
            break;
          }
      for (types::global_dof_index jj=0; jj<helper[0].size(); ++jj)
        if (helper[1][jj].is_element(2 * (jb1 + jb2*nf1 + jb3*nf1*nf2)))
          {
            grid_sets[1][jj].add_index(j);
            break;
          }

    }
  // We have some problems in this compress.
  // for(types::global_dof_index i=0; i<grid_sets.size(); ++i)
  //   for(types::global_dof_index j=0; i<grid_sets[i].size(); ++j)
  //     grid_sets[i][j].compress();

}
// We compute all the operations linked to the requested tolerance.
// In particular the tolerance sets the number of points in the fine grid,
// and the spread of the Gaussian convolution (nspread).
// We have chosen to maintain Greengard's optimisations by means of unlooped
// for cycles.
void BlackNUFFT::compute_tolerance_infos()
{
  TimerOutput::Scope t(computing_timer, " Tolerance Infos ");

  //Now we compute all the stuff starting from epsilon

  //It is the oversampling RATio. Mr/M = (fine grid point with oversampling) / (fine grid point)
  // the more accuracy you require the more you have to oversample.

  if (gridding == "FGG")
    {
      if (epsilon <= 1e-12)
        rat = 3.;
      else if (epsilon <= 1e-11)
        rat = std::sqrt(3.3);
      else
        rat = std::sqrt(2.2);

      // int(-log(eps)/(pi*(rat-1d0)/(rat-.5d0)) + .5d0)
      nspread = int(-std::log(epsilon)/(numbers::PI*(rat-1.)/(rat-.5)) + .5);

      // Grid span parameters
      double t1 = 2. * xm[0] * sm[0] / numbers::PI;
      double t2 = 2. * xm[1] * sm[1] / numbers::PI;
      double t3 = 2. * xm[2] * sm[2] / numbers::PI;

      // pcout<<rat<<" "<<xm[2]<<" "<<sm[2]<<std::endl;

      // Oversampled grid sizes.
      nf1 = (types::global_dof_index)next235(rat*std::max(rat*t1+2*nspread,2*nspread/(rat-1)));
      nf2 = (types::global_dof_index)next235(rat*std::max(rat*t2+2*nspread,2*nspread/(rat-1)));
      nf3 = (types::global_dof_index)next235(rat*std::max(rat*t3+2*nspread,2*nspread/(rat-1)));


      // Oversampling parameters in the 3 directions.
      rat1 = (std::sqrt(nf1*t1+nspread*nspread)-nspread)/t1;
      rat2 = (std::sqrt(nf2*t2+nspread*nspread)-nspread)/t2;
      rat3 = (std::sqrt(nf3*t3+nspread*nspread)-nspread)/t3;


      r2lamb1 = rat1 * rat1 * nspread / (rat1*(rat1-0.5));
      r2lamb2 = rat2 * rat2 * nspread / (rat2*(rat2-0.5));
      r2lamb3 = rat3 * rat3 * nspread / (rat3*(rat3-0.5));


      hx = numbers::PI/(rat1*sm[0]);
      hs = double(2)*numbers::PI/double(nf1)/hx;
      hy = numbers::PI/(rat2*sm[1]);
      ht = double(2)*numbers::PI/double(nf2)/hy;
      hz = numbers::PI/(rat3*sm[2]);
      // pcout<<xm[2]<<std::endl;
      hu = double(2)*numbers::PI/double(nf3)/hz;

      iw7 = (types::global_dof_index)(nf1*(r2lamb1-nspread)/r2lamb1+.1);
      iw8 = (types::global_dof_index)(nf2*(r2lamb2-nspread)/r2lamb2+.1);
      iw9 = (types::global_dof_index)(nf3*(r2lamb3-nspread)/r2lamb3+.1);

      // pcout<<epsilon<<" "<<nspread<<" "<<nf1<<" "<<nf2<<" "<<nf3<<std::endl;
      // pcout<<iw7<<" "<<iw8<<" "<<iw9<<" "<<std::endl;


      double t4 = numbers::PI * r2lamb1 / double(nf1*nf1);
      double cross1 = (1. - 2. * (nf1/2 % 2)) / r2lamb1;
      deconv_array_x.reinit(iw7+1);
      for ( types::global_dof_index k1 = 0; k1 <= iw7; ++k1)
        {
          deconv_array_x[k1] = cross1*std::exp(t4*double(k1*k1));
          cross1 = -cross1;
        }
      double t5 = numbers::PI * r2lamb2 / double(nf2*nf2);
      cross1 = 1./r2lamb2;
      deconv_array_y.reinit(iw8+1);
      for ( types::global_dof_index k1 = 0; k1 <= iw8; ++k1)
        {
          deconv_array_y[k1] = cross1*std::exp(t5*double(k1*k1));
          cross1 = -cross1;
        }
      double t6 = numbers::PI * r2lamb3 / double(nf3*nf3);
      cross1 = 1./r2lamb3;
      deconv_array_z.reinit(iw9+1);
      for ( types::global_dof_index k1 = 0; k1 <= iw9; ++k1)
        {
          deconv_array_z[k1] = cross1*std::exp(t6*k1*k1);
          cross1 = -cross1;
        }

    }
  else
    AssertThrow(true, ExcNotImplemented());


}

// This functions computes the bounding box of the coarse grids (input and output).
// For the sake of simplicity we have chosen to make a loop over the dimensions.
// We need both the box spans and midpoints in order to perform all the FFT computations
// in the interval around the origin and then translate it back.

void BlackNUFFT::compute_ranges()
{
  TimerOutput::Scope t(computing_timer, " Compute Ranges ");

  // Loop over input grid
  for (unsigned int i=0; i<3; ++i)
    {
      double t1 = input_grid[i][0];
      double t2 = input_grid[i][0];
      for ( unsigned int j = 1; j < nj; ++j)
        {
          if (input_grid[i][j] > t2)
            t2 = input_grid[i][j];
          else if (input_grid[i][j] < t1)
            t1 = input_grid[i][j];
        }
      xb[i] = (t1+t2) / 2.;
      xm[i] = std::max(t2-xb[i],-t1+xb[i]);  //max(abs(t2-xb),abs(t1-xb))
    }
  // Loop over output grid
  for (unsigned int i=0; i<3; ++i)
    {
      double t1 = output_grid[i][0];
      double t2 = output_grid[i][0];
      for ( unsigned int j = 1; j < nk; ++j)
        {
          if (output_grid[i][j] > t2)
            t2 = output_grid[i][j];
          else if (output_grid[i][j] < t1)
            t1 = output_grid[i][j];
        }
      sb[i] = (t1+t2) / 2.;
      sm[i] = std::max(t2-sb[i],-t1+sb[i]);  //max(abs(t2-xb),abs(t1-xb))
    }
}


void BlackNUFFT::input_gridding()
{
  TimerOutput::Scope t(computing_timer, " Input Gridding ");

  if (gridding == "FGG")
    {
      fast_gaussian_gridding_on_input();
      scaling_input_gridding();
    }
  else if (gridding == "MINMAX")
    {
      AssertThrow(true, ExcNotImplemented())
    }
}

void BlackNUFFT::output_gridding()
{
  TimerOutput::Scope t(computing_timer, " Output Gridding ");

  if (gridding == "FGG")
    {
      fast_gaussian_gridding_on_output();
      scaling_output_gridding();
    }
  else if (gridding == "MINMAX")
    {
      AssertThrow(true, ExcNotImplemented())
    }
}

// This function performs the initial Gaussian gridding. We have chosen to maintain
// Greengard's fast implementation of the Gauss function. This is essential since
// this function and its counterpart are very computationally expensive.
void BlackNUFFT::fast_gaussian_gridding_on_input()
{
  TimerOutput::Scope t(computing_timer, " Fast Gaussian Gridding on Inputs ");

  // Three handle variables
  double t1 = numbers::PI / r2lamb1;
  double t2 = numbers::PI / r2lamb2;
  double t3 = numbers::PI / r2lamb3;


  // These three vector only store the values of the exponential

  xexp.reinit(nspread);
  yexp.reinit(nspread);
  zexp.reinit(nspread);

  for (unsigned int k1 = 0; k1 < nspread; ++k1)
    {
      xexp[k1] = std::exp(-t1*(k1+1)*(k1+1));
      yexp[k1] = std::exp(-t2*(k1+1)*(k1+1));
      zexp[k1] = std::exp(-t3*(k1+1)*(k1+1));
    }

  if (!fft_backward)
    {
      sb[0] = -sb[0];
      sb[1] = -sb[1];
      sb[2] = -sb[2];
    }

  // In the following we use WorkStream to parallelise, through TBB, the setting up
  // of the initial preconditioner that does not consider any constraint.
  // We define two structs that are needed: the first one is empty since we have decided to use
  // the capture of lambda functions to let the worker know what it needs. The second one
  // instead is filled by each worker and passed down by reference to the copier that manage any racing conditions
  // copying properly the computed data where they belong.
  struct FGGScratch {};

  // The copier structure holds the thing needed to compute the actual position on the global array
  // for the copy operation.
  struct FGGCopy
  {
    Vector<double> local_fine_grid_data;
    // Variables needed for the Fast Gaussian evaluation.
    double diff1, diff2, diff3, ang;
    types::global_dof_index jb1, jb2, jb3;
    std::complex<double> cs;
    double cross, cross1;
    std::vector<types::global_dof_index> local_to_global;

  };

  // The worker function uses the capture to know the actual state of the BlackNUFFT class.
  // In this way we can perform the computation
  // of the column to be added at each row quite straigtforwardly. Since all the
  // workers must be able to run in parallel we must be sure that no racing condition occurs.
  auto f_fgg_worker = [this,t1,t2,t3] (IndexSet::ElementIterator j_it, FGGScratch &foo, FGGCopy &copy_data)
  {
    types::global_dof_index j=*j_it;
    // We resize everything to be sure to compute, and then copy only the needed data.
    copy_data.local_fine_grid_data.reinit(2*2*nspread*2*nspread*2*nspread);
    copy_data.local_to_global.resize(2*nspread*2*nspread*2*nspread);
    // Vectors needed for the precomputations along the three dimensions. In principle they
    // belong here but in maybe in the copier they perform better.
    Vector<double> xc(2*nspread), yc(2*nspread), zc(2*nspread);

    copy_data.jb1 = types::global_dof_index(double(nf1/2) + (input_grid[0][j]-xb[0])/hx);
    copy_data.jb2 = types::global_dof_index(double(nf2/2) + (input_grid[1][j]-xb[1])/hy);
    copy_data.jb3 = types::global_dof_index(double(nf3/2) + (input_grid[2][j]-xb[2])/hz);
    copy_data.diff1 = double(nf1/2) + (input_grid[0][j]-xb[0])/hx - copy_data.jb1;
    copy_data.diff2 = double(nf2/2) + (input_grid[1][j]-xb[1])/hy - copy_data.jb2;
    copy_data.diff3 = double(nf3/2) + (input_grid[2][j]-xb[2])/hz - copy_data.jb3;
    copy_data.ang = sb[0]*input_grid[0][j] + sb[1]*input_grid[1][j] + sb[2]*input_grid[2][j];
    std::complex<double> dummy1(std::cos(copy_data.ang), std::sin(copy_data.ang));
    std::complex<double> dummy2(input_vector[2*j], input_vector[2*j+1]);
    copy_data.cs = dummy1 * dummy2;
    // 2) We precompute everything along x. Fast Gaussian Gridding
    // 2a) Precomptaiton in x
    // The original loop was -nspread+1 : nspread
    xc[nspread-1] = std::exp(-t1*copy_data.diff1*copy_data.diff1
                             -t2*copy_data.diff2*copy_data.diff2
                             -t3*copy_data.diff3*copy_data.diff3);

    copy_data.cross = xc[nspread-1];
    copy_data.cross1 = exp(2.*t1 * copy_data.diff1);
    for (unsigned int k1 = 0; k1 < nspread; ++k1)
      {
        copy_data.cross = copy_data.cross * copy_data.cross1;
        xc[nspread+k1] = xexp[k1]*copy_data.cross;
      }
    copy_data.cross = xc[nspread-1];
    copy_data.cross1 = 1./copy_data.cross1;
    for (unsigned int k1 = 0; k1 < nspread-1; ++k1) // Precomputing everything Watch out for negative indices.
      {
        copy_data.cross = copy_data.cross * copy_data.cross1;
        xc[nspread-k1-2] = xexp[k1]*copy_data.cross;
      }
    // 2b) Precomptaiton in y
    yc[nspread-1] = 1.;
    copy_data.cross = std::exp(2.*t2 * copy_data.diff2);
    copy_data.cross1 = copy_data.cross;
    for (unsigned int k2 = 0; k2 < nspread-1; ++k2) //k2 = 1, nspread-1
      {
        yc[nspread + k2] = yexp[k2]*copy_data.cross;
        yc[nspread-2-k2] = yexp[k2]/copy_data.cross;
        copy_data.cross = copy_data.cross * copy_data.cross1;
      }
    yc[2*nspread-1] = yexp[nspread-1]*copy_data.cross;
    // 2c) Precomptaiton in z
    zc[nspread-1] = 1.;
    copy_data.cross = std::exp(2.*t3 * copy_data.diff3);
    copy_data.cross1 = copy_data.cross;
    for (unsigned int k3 = 0; k3 < nspread-1; ++k3)
      {
        zc[nspread + k3] = zexp[k3]*copy_data.cross;
        zc[nspread-2-k3] = zexp[k3]/copy_data.cross;
        copy_data.cross = copy_data.cross * copy_data.cross1;
      }
    zc[2*nspread-1] = zexp[nspread-1]*copy_data.cross;
    // 2d) We put everything together locally
    for (unsigned int k3 = 0; k3<2*nspread; ++k3)
      {
        std::complex<double> c2;
        c2 = zc[k3] * copy_data.cs;
        for (unsigned int k2 = 0; k2<2*nspread; ++k2)
          {
            std::complex<double> cc;
            cc = yc[k2] * c2;
            types::global_dof_index ii = copy_data.jb1 + (copy_data.jb2+k2-(nspread-1))*nf1 + (copy_data.jb3+k3-(nspread-1))*nf1*nf2;
            for (unsigned int k1 = 0; k1<2*nspread; ++k1)
              {
                types::global_dof_index istart = 2*(ii+((int)k1 - (int)(nspread-1)));
                std::complex<double> zz;
                zz = xc[k1] * cc;
                unsigned int local_index = 2*(k3*(2*nspread*2*nspread)+k2*(2*nspread)+k1);
                copy_data.local_fine_grid_data[local_index] += zz.real();
                copy_data.local_fine_grid_data[local_index+1] += zz.imag();
                copy_data.local_to_global[local_index/2] = istart;
              }
          }
      }

  };

  // The copier function uses the InitPrecCopy structure to know the global indices to add to
  // the global initial sparsity pattern. We use once again the capture to access the global memory.
  auto f_fgg_copier = [this] (const FGGCopy &copy_data)
  {
    if (fftw3_set.is_element(2 * (copy_data.jb1 + copy_data.jb2*nf1 + copy_data.jb3*nf1*nf2)))
      {

        auto fgg_putter = [](unsigned int k3, parallel::distributed::Vector<double> &copy_fine_grid_data, const FGGCopy &copy_data, const BlackNUFFT * foo_nufft)
        {
          unsigned int local_index = 2*(k3*(2*foo_nufft->nspread*2*foo_nufft->nspread));
          for (unsigned int k2 = 0; k2<2*foo_nufft->nspread; ++k2)
            {
              for (unsigned int k1 = 0; k1<2*foo_nufft->nspread; ++k1)
                {
                  // types::global_dof_index istart = 2*(ii+(k1 - (nspread-1)));
                  // OCCHIO AL CONTATORE
                  // unsigned int local_index = 2*(k3*(2*foo_nufft->nspread*2*foo_nufft->nspread)+k2*(2*foo_nufft->nspread)+k1);
                  copy_fine_grid_data[copy_data.local_to_global[local_index/2]] += copy_data.local_fine_grid_data[local_index];
                  copy_fine_grid_data[copy_data.local_to_global[local_index/2]+1] += copy_data.local_fine_grid_data[local_index+1];
                  local_index += 2;
                }
            }

        };

        // For any k3 we have no race condition so we can use TaskGroup to handle it
        Threads::TaskGroup<> group_fgg_putter;
        for (unsigned int k3 = 0; k3<2*nspread; ++k3)
          {

            unsigned int local_index = 2*(k3*(2*nspread*2*nspread));
            for (unsigned int k2 = 0; k2<2*nspread; ++k2)
              {
                for (unsigned int k1 = 0; k1<2*nspread; ++k1)
                  {
                    // types::global_dof_index istart = 2*(ii+(k1 - (nspread-1)));
                    // OCCHIO AL CONTATORE
                    // unsigned int local_index = 2*(k3*(2*foo_nufft->nspread*2*foo_nufft->nspread)+k2*(2*foo_nufft->nspread)+k1);
                    fine_grid_data[copy_data.local_to_global[local_index/2]] += copy_data.local_fine_grid_data[local_index];
                    fine_grid_data[copy_data.local_to_global[local_index/2]+1] += copy_data.local_fine_grid_data[local_index+1];
                    local_index += 2;
                  }
              }

            // group_fgg_putter += Threads::new_task ( static_cast<void (*)(unsigned int, parallel::distributed::Vector<double> &, const FGGCopy &, const BlackNUFFT *)> (fgg_putter), k3, fine_grid_data, copy_data, this);
          }
        // group_fgg_putter.join_all();
      }

  };

  unsigned int foo1,foo2;
  auto f_dummy_worker_odd = [f_fgg_worker,f_fgg_copier,this](types::global_dof_index first_it, unsigned int foo1, unsigned int foo2)
  {
    // We need to create two empty structures that will be copied by WorkStream and passed
    // to each worker-copier to compute the sparsity pattern for blocks in the childlessList.
    FGGScratch foo_scratch;
    FGGCopy foo_copy;
    WorkStream::run(grid_sets[0][first_it].begin(), grid_sets[0][first_it].end(),
                    f_fgg_worker, f_fgg_copier, foo_scratch, foo_copy);
  };
  auto f_dummy_copier_odd = [](const unsigned int foo2) {};
  auto f_dummy_worker_even = [f_fgg_worker,f_fgg_copier,this](types::global_dof_index first_it, unsigned int foo1, unsigned int foo2)
  {
    FGGScratch foo_scratch;
    FGGCopy foo_copy;
    WorkStream::run(grid_sets[1][first_it].begin(), grid_sets[1][first_it].end(),
                    f_fgg_worker, f_fgg_copier, foo_scratch, foo_copy);
  };
  auto f_dummy_copier_even = [](const unsigned int foo2) {};

  auto f_dummy_worker_even_tbb = [f_fgg_worker,f_fgg_copier,this](blocked_range<unsigned int> r)
  {
    for (unsigned int i=r.begin(); i<r.end(); ++i)
      {
        auto indy = grid_sets[1][i];
        FGGScratch foo_scratch;
        FGGCopy foo_copy;
        WorkStream::run(indy.begin(), indy.end(),
                        f_fgg_worker, f_fgg_copier, foo_scratch, foo_copy, 1, 1);

      }
  };

  auto f_dummy_worker_odd_tbb = [f_fgg_worker,f_fgg_copier,this](blocked_range<unsigned int> r)
  {
    for (unsigned int i=r.begin(); i<r.end(); ++i)
      {
        auto indy = grid_sets[0][i];
        FGGScratch foo_scratch;
        FGGCopy foo_copy;
        WorkStream::run(indy.begin(), indy.end(),
                        f_fgg_worker, f_fgg_copier, foo_scratch, foo_copy, 1, 1);

      }
  };


  parallel_for(blocked_range<unsigned int> (0, grid_sets[1].size(),1), f_dummy_worker_even_tbb);

  parallel_for(blocked_range<unsigned int> (0, grid_sets[0].size(),1), f_dummy_worker_odd_tbb);


  // WorkStream::run(0,grid_sets[1].size(),f_dummy_worker_even,f_dummy_copier_even,foo1,foo2);//,2*MultithreadInfo::n_threads(),8);
  //
  // WorkStream::run(0,grid_sets[0].size(),f_dummy_worker_odd,f_dummy_copier_odd,foo1,foo2);//,2*MultithreadInfo::n_threads(),8);


  // The following is to have the classical WorkStream run on all the input set
  // WorkStream::run(input_set.begin(), input_set.end(), f_fgg_worker, f_fgg_copier, foo_scratch, foo_copy);
  fine_grid_data.compress(VectorOperation::add);
  // pcout<<fine_grid_data.l2_norm()<<std::endl;


  if (!fft_backward)
    {
      sb[0] = -sb[0];
      sb[1] = -sb[1];
      sb[2] = -sb[2];
    }

}

// The following functions performs the deconvolution to correct the second
// Gaussian gridding. This should be just a pointwise multiplication. For this
// reason we can use TaskGroup withuot caring about racing conditions.

void BlackNUFFT::scaling_input_gridding()
{
  TimerOutput::Scope t(computing_timer, " Deconvolution Before FFT ");


  auto f_scaling_input_gridding = [] (types::global_dof_index k2, types::global_dof_index k1, parallel::distributed::Vector<double> &fine_grid_data_copy, const BlackNUFFT *foo_nufft)
  {

    types::global_dof_index ii;
    ii = (foo_nufft->nf1/2+k1-foo_nufft->iw7) +
         (foo_nufft->nf2/2+k2-foo_nufft->iw8)*foo_nufft->nf1 +
         (foo_nufft->nf3/2)*foo_nufft->nf1*foo_nufft->nf2;

    double cross = foo_nufft->deconv_array_x[std::abs((types::signed_global_dof_index)k1-(types::signed_global_dof_index)foo_nufft->iw7)] *
                   foo_nufft->deconv_array_y[std::abs((types::signed_global_dof_index)k2-(types::signed_global_dof_index)foo_nufft->iw8)];

    std::complex<double> c2;
    std::complex<double> zz;

    if (foo_nufft->fftw3_set.is_element(2*ii))
      {
        c2 = std::complex<double>(fine_grid_data_copy[2*ii],fine_grid_data_copy[2*ii+1]);
        zz = (cross*foo_nufft->deconv_array_z[0])*c2;
        fine_grid_data_copy[2*ii] = zz.real();
        fine_grid_data_copy[2*ii+1] = zz.imag();
      }

    for (types::global_dof_index k3 = 1; k3 <= foo_nufft->iw9; ++k3)
      {

        types::global_dof_index is2;

        is2 = 2*(ii+k3*foo_nufft->nf1*foo_nufft->nf2);
        if (foo_nufft->fftw3_set.is_element(is2))
          {
            c2 = std::complex<double>(fine_grid_data_copy[is2],fine_grid_data_copy[is2+1]);
            zz = (cross*foo_nufft->deconv_array_z[k3])*c2;
            fine_grid_data_copy[is2] = zz.real();
            fine_grid_data_copy[is2+1] = zz.imag();
          }
        is2 = 2*(ii-k3*foo_nufft->nf1*foo_nufft->nf2);
        if (foo_nufft->fftw3_set.is_element(is2))
          {
            c2 = std::complex<double>(fine_grid_data_copy[is2],fine_grid_data_copy[is2+1]);
            zz = (cross*foo_nufft->deconv_array_z[k3])*c2;
            fine_grid_data_copy[is2] = zz.real();
            fine_grid_data_copy[is2+1] = zz.imag();
          }
        // std::cout<<is2<<" ";

      }
  };

  auto f_scaling_input_gridding_tbb = [this] (blocked_range<types::global_dof_index> r)
  {
    for (types::global_dof_index k2=r.begin(); k2<r.end(); ++k2)
      {
        // std::cout<<"k2 : "<<k2<<std::endl;
        for (types::global_dof_index k1=0; k1<2*iw7+1; ++k1)
          {
            types::global_dof_index ii;
            ii = ( nf1/2+k1- iw7) +
                 ( nf2/2+k2- iw8)* nf1 +
                 ( nf3/2)* nf1* nf2;

            double cross =  deconv_array_x[std::abs((types::signed_global_dof_index)k1-(types::signed_global_dof_index) iw7)] *
                            deconv_array_y[std::abs((types::signed_global_dof_index)k2-(types::signed_global_dof_index) iw8)];

            std::complex<double> c2;
            std::complex<double> zz;

            if ( fftw3_set.is_element(2*ii))
              {
                c2 = std::complex<double>(fine_grid_data[2*ii],fine_grid_data[2*ii+1]);
                zz = (cross* deconv_array_z[0])*c2;
                fine_grid_data[2*ii] = zz.real();
                fine_grid_data[2*ii+1] = zz.imag();
              }

            for (types::global_dof_index k3 = 1; k3 <=  iw9; ++k3)
              {

                types::global_dof_index is2;

                is2 = 2*(ii+k3* nf1* nf2);
                if ( fftw3_set.is_element(is2))
                  {
                    c2 = std::complex<double>(fine_grid_data[is2],fine_grid_data[is2+1]);
                    zz = (cross* deconv_array_z[k3])*c2;
                    fine_grid_data[is2] = zz.real();
                    fine_grid_data[is2+1] = zz.imag();
                  }
                is2 = 2*(ii-k3* nf1* nf2);
                if ( fftw3_set.is_element(is2))
                  {
                    c2 = std::complex<double>(fine_grid_data[is2],fine_grid_data[is2+1]);
                    zz = (cross* deconv_array_z[k3])*c2;
                    fine_grid_data[is2] = zz.real();
                    fine_grid_data[is2+1] = zz.imag();
                  }
                // std::cout<<is2<<" ";
              }
          }
        // std::cout<<std::endl;
      }
  };

  Threads::TaskGroup<> scaling_input_gridding_group;
  parallel_for(blocked_range<types::global_dof_index> (0, 2*iw8+1,10), f_scaling_input_gridding_tbb);

  // for (types::global_dof_index k2 = 0; k2<2*iw8+1; ++k2)
  //   {
  //     // std::cout<<"k2 : "<<k2<<std::endl;
  //     for (types::global_dof_index k1 = 0; k1<2*iw7+1; ++k1)
  //       {
  //         scaling_input_gridding_group += Threads::new_task ( static_cast<void (*)(types::global_dof_index, types::global_dof_index, parallel::distributed::Vector<double> &, const BlackNUFFT *)> (f_scaling_input_gridding), k2, k1, fine_grid_data, this);
  //         // f_scaling_input_gridding(k2,k1,fine_grid_data,this);
  //       }
  //       // std::cout<<std::endl;
  //
  //   }
  // scaling_input_gridding_group.join_all();

  // for(types::global_dof_index i = 0; i<fine_grid_data.size()/2; ++i)
  //   // fine_grid_data[i]=test_data_before[i];
  //   if(std::abs(fine_grid_data[2*i]-test_data_before[2*i])>1e-3)
  //   {
  //     std::cout<<"ERROR BEFORE FFT "<<fine_grid_data[2*i]<<" "<<test_data_before[2*i]<<" "<<fine_grid_data[2*i+1]<<" "<<test_data_before[2*i+1]<<" "<<2*i<<std::endl;
  //   }


}


// TO BE REMOVED
void BlackNUFFT::shift_data_before_fft()
{
  TimerOutput::Scope t(computing_timer, " Shift before FFT ");

  // fine_grid_data.print(std::cout);
  // fine_grid_data = 0.;
  //   for(types::global_dof_index k3=0; k3<2*(iw9+nspread)+1; ++k3)
  //   {
  //     for(types::global_dof_index k2=0; k2<2*(iw8+nspread)+1; ++k2)
  //     {
  //       for(types::global_dof_index k1=0; k1<2*(iw7+nspread)+1; ++k1)
  //       {
  //           types::global_dof_index old_pos = 2*((nf3/2 + k3 - iw9 - nspread)*nf1*nf2 +
  //                                             (nf2/2 + k2 - iw8 - nspread)*nf1 +
  //                                             (nf1/2 + k1 - iw7 - nspread));
  //           fine_grid_data[old_pos] = 1.;//fine_grid_data[new_pos];
  //         fine_grid_data[old_pos+1] = 1.;//fine_grid_data[new_pos+1];
  //
  //       }
  //     }
  //   }

  // THIS SEEMS TO BE ALMOST GOOD IF BACKWARD, ELSE USE -1^(i+j+k)
  // std::cout<<"pippo"<<std::endl;
  // for(types::global_dof_index k3=nf3/2; k3<nf3/2+1; k3=k3+1)
  // {
  //   std::cout<<fine_grid_data[2*(k3*nf1*nf2 + nf2/2*nf1 + nf1/2)+1]<<std::endl;
  //
  //   for(types::global_dof_index k2=0; k2<nf2; ++k2)
  //   {
  //     for(types::global_dof_index k1=0; k1<nf1; ++k1)
  //     {
  //       types::global_dof_index old_pos = 2*(k3*nf1*nf2 +
  //                                         k2*nf1 +
  //                                         k1)-1;
  //
  //       std::cout<<fine_grid_data[old_pos+1]<<" ";
  //     }
  //     std::cout<<std::endl;
  //   }
  // }

  for (types::global_dof_index k3=0; k3<(nf3)/2; ++k3)
    {
      types::global_dof_index new_pos_3;
      if (k3 >= nf3/2)
        new_pos_3 = k3-nf3/2;
      else
        new_pos_3 = nf3/2 +k3;
      if (k3 == 6)
        std::cout<<new_pos_3<<std::endl;
      for (types::global_dof_index k2=0; k2<(nf2); ++k2)
        {
          types::global_dof_index new_pos_2;
          if (k2 >= nf2/2)
            new_pos_2 = k2-nf2/2;
          else
            new_pos_2 = nf2/2 +k2;
          for (types::global_dof_index k1=0; k1<(nf1); ++k1)
            {
              types::global_dof_index new_pos_1;
              if (k1 >= nf1/2)
                new_pos_1 = k1-nf1/2;
              else
                new_pos_1 = nf1/2 +k1;

              types::global_dof_index old_pos = 2*((k3)*nf1*nf2 +
                                                   (k2)*nf1 +
                                                   (k1));
              types::global_dof_index new_pos = 2*((new_pos_3)*nf1*nf2 +
                                                   (new_pos_2)*nf1 +
                                                   (new_pos_1));

              // std::cout<<k3<<" "<<k2<<" "<<k1<<" "<<old_pos<<" "<<std::endl;
              double foo_real = fine_grid_data [old_pos];
              double foo_imag = fine_grid_data [old_pos+1];
              // if(k3 == 6 && foo_imag != 0.)
              //   std::cout<<fine_grid_data[old_pos+1]<<std::endl;
              fine_grid_data[old_pos] = fine_grid_data[new_pos];
              fine_grid_data[old_pos+1] = fine_grid_data[new_pos+1];
              fine_grid_data[new_pos] = foo_real;
              fine_grid_data[new_pos+1] = foo_imag;
              // if(k3 == 6 && foo_imag != 0.)
              //   std::cout<<old_pos<<" "<<foo_imag<<" "<<new_pos<<" "<<fine_grid_data[old_pos+1]<<std::endl;
              // if(k3 == 6 && foo_imag != 0.)
              //   std::cout<<fine_grid_data[old_pos+1]<<std::endl;

            }
        }
    }
  // std::cout<<"pippo"<<std::endl;
  //
  // for(types::global_dof_index k3=16; k3<17; k3=k3+1)
  // {
  //   for(types::global_dof_index k2=0; k2<nf2; ++k2)
  //   {
  //     for(types::global_dof_index k1=0; k1<nf1; ++k1)
  //     {
  //       types::global_dof_index old_pos = 2*(k3*nf1*nf2 +
  //                                         k2*nf1 +
  //                                         k1);
  //
  //       std::cout<<fine_grid_data[old_pos+1]<<" ";
  //     }
  //     std::cout<<std::endl;
  //   }
  // }


}


// This function computes the 3d fft on the fine array. We have chosen to use
// the state of the art FFTW library.
void BlackNUFFT::compute_fft_3d()
{
  TimerOutput::Scope t(computing_timer, " 3D FFTW ");
  if (fft_type == "FFTW")
    {
      fftw_plan p;
      fftw_complex *dummy;

      // We need a cast to make FFTW accept the lacal double array as a complex one.
      // We have taken care of compatibility before so we just need a cast.
      dummy = reinterpret_cast<fftw_complex *> (&fine_grid_data.local_element(0));

      // We don't need the ghost cells set up by the gridding anymore so we wipe them
      // out.
      fine_grid_data.zero_out_ghosts();
      if (fft_backward)
        {
          p = fftw_mpi_plan_dft_3d(nf3, nf2, nf1, dummy, dummy, mpi_communicator, FFTW_BACKWARD, FFTW_ESTIMATE);
          fftw_execute(p);
          pcout<<"BACKWARD FFT"<<std::endl;
          fftw_destroy_plan(p);
        }
      else
        {
          p = fftw_mpi_plan_dft_3d(nf3, nf2, nf1,dummy, dummy, mpi_communicator, FFTW_FORWARD, FFTW_ESTIMATE);
          fftw_execute(p);
          pcout<<"FORWARD FFT"<<std::endl;
          fftw_destroy_plan(p);

        }
    }
  else
    {
      AssertThrow(true, ExcNotImplemented());
    }

}

// The algorithm is based on a shifted FFT. For this reason we apply a circular
// shifting on the transformed fine grid to obtain the shift. Basically we just need
// to multiply each element by -1^(i+j+k). This is a local operation so we can use
// TaskGroup withot caring on synchronisation.
// TODO: Find a smarter way than pow(-1,i+j+k).
void BlackNUFFT::shift_data_for_fftw3d()
{
  TimerOutput::Scope t(computing_timer, " Shift Data for FFTW3D ");


  auto f_shift_odd_start = [] (types::global_dof_index k3, parallel::distributed::Vector<double> &fine_grid_data_copy, const BlackNUFFT *foo_nufft)
  {

    for (types::global_dof_index k2 = 0; k2 < (foo_nufft->nf2)*foo_nufft->nf1; k2=k2+2)
      {
        fine_grid_data_copy[2*(k3*foo_nufft->nf1*foo_nufft->nf2+k2)] *= -1;
        fine_grid_data_copy[2*(k3*foo_nufft->nf1*foo_nufft->nf2+k2)+1] *= -1;

      }
  };
  auto f_shift_even_start = [] (types::global_dof_index k3, parallel::distributed::Vector<double> &fine_grid_data_copy, const BlackNUFFT *foo_nufft)
  {

    for (types::global_dof_index k2 = 0; k2 < foo_nufft->nf2; k2=k2+1)
      {
        for (types::global_dof_index k1 = 0; k1 < (foo_nufft->nf1); ++k1)
          {

            fine_grid_data_copy[2*(k3*foo_nufft->nf1*foo_nufft->nf2+k2)] *= -1;
            fine_grid_data_copy[2*(k3*foo_nufft->nf1*foo_nufft->nf2+k2)+1] *= -1;

          }
      }
  };
  auto f_shift = [] (types::global_dof_index k3, parallel::distributed::Vector<double> &fine_grid_data_copy, const BlackNUFFT *foo_nufft)
  {
    for (types::global_dof_index k2 = 0; k2 < (foo_nufft->nf2); ++k2)
      {
        // types::global_dof_index ii = (nf2/2+k2-nspread-iw8)*nf1 + (nf3/2+k3-nspread-iw9)*nf1*nf2;
        for (types::global_dof_index k1 = 0; k1 < (foo_nufft->nf1); ++k1)
          {
            // types::global_dof_index istart = 2 * (ii+nf1/2+k1);
            // types::global_dof_index is2 = 2 * (ii+nf1/2-k1);
            // fine_grid_data[istart-1] = - fine_grid_data[istart-1];
            // fine_grid_data[istart+1-1] = - fine_grid_data[istart+1-1];
            // fine_grid_data[is2-1] = - fine_grid_data[is2-1];
            // fine_grid_data[is2+1-1] = - fine_grid_data[is2+1-1];

            double multi = -2*(double)((k3+k2+k1)%2)+1;
            // std::cout<<multi<<" "<<std::pow(-1,k3+k2+k1) <<std::endl;
            fine_grid_data_copy[2*(k3*foo_nufft->nf1*foo_nufft->nf2+k2*foo_nufft->nf1+k1)] *= multi;//std::pow(-1,k3+k2+k1);
            fine_grid_data_copy[2*(k3*foo_nufft->nf1*foo_nufft->nf2+k2*foo_nufft->nf1+k1)+1] *= multi;//std::pow(-1,k3+k2+k1);
            // fine_grid_data_copy[2*(k3*foo_nufft->nf1*foo_nufft->nf2+k2*foo_nufft->nf1+k1)] -= 2*fine_grid_data_copy[2*(k3*foo_nufft->nf1*foo_nufft->nf2+k2*foo_nufft->nf1+k1)];
            // fine_grid_data_copy[2*(k3*foo_nufft->nf1*foo_nufft->nf2+k2*foo_nufft->nf1+k1)+1] -= 2*fine_grid_data_copy[2*(k3*foo_nufft->nf1*foo_nufft->nf2+k2*foo_nufft->nf1+k1)+1];


          }

      }
  };

  auto f_shift_tbb = [this] (const blocked_range<types::global_dof_index> &r)
  {
    for (types::global_dof_index k3=r.begin(); k3<r.end(); ++k3)
      for (types::global_dof_index k2 = 0; k2 < (nf2); ++k2)
        {
          // types::global_dof_index ii = (nf2/2+k2-nspread-iw8)*nf1 + (nf3/2+k3-nspread-iw9)*nf1*nf2;
          for (types::global_dof_index k1 = 0; k1 < (nf1); ++k1)
            {
              double multi = -2*(double)((k3+k2+k1)%2)+1;
              fine_grid_data[2*(k3*nf1*nf2+k2*nf1+k1)] *= multi;//std::pow(-1,k3+k2+k1);
              fine_grid_data[2*(k3*nf1*nf2+k2*nf1+k1)+1] *= multi;//std::pow(-1,k3+k2+k1);


            }

        }
  };


  Threads::TaskGroup<> shift_data_group;
  // We need the shift only on the locally owned data.
  types::global_dof_index blocking=10;
  tbb::parallel_for(blocked_range<types::global_dof_index> (local_nf3_start, local_nf3+local_nf3_start,blocking), f_shift_tbb);
  // for (types::global_dof_index k3 = local_nf3_start; k3<(local_nf3+local_nf3_start); ++k3)
  //   {
  //     shift_data_group += Threads::new_task ( static_cast<void (*)(types::global_dof_index, parallel::distributed::Vector<double> &, const BlackNUFFT *)> (f_shift), k3, fine_grid_data, this);
  //   }
  //
  // shift_data_group.join_all();

  // Now we can distribute the results of the shifted 3d FFT.
  fine_grid_data.compress(VectorOperation::add);
  // We need to update the ghost values for a proper fast gaussian
  // gridding on the output array.
  fine_grid_data.update_ghost_values();
  // pcout<<fine_grid_data.l2_norm()<<std::endl;

}

// We need a second Fast Gaussian Gridding to recover the result on the desired points.
// There should not be much synchronisations in this algorithm, therefore we can use straightforwardly
// WorkStream on the output set without splitting the grid.
void BlackNUFFT::fast_gaussian_gridding_on_output()
{
  TimerOutput::Scope t(computing_timer, " Fast Guassian Gridding on Output ");
  double t1 = numbers::PI/r2lamb1;
  double t2 = numbers::PI/r2lamb2;
  double t3 = numbers::PI/r2lamb3;

  for (types::global_dof_index i = 0 ; i<2*nk; ++i)
    output_vector[i] = 0.;
  // In the following we use WorkStream to parallelise, through TBB, the setting up
  // of the initial preconditioner that does not consider any constraint.
  // We define two structs that are needed: the first one is empty since we have decided to use
  // the capture of lambda functions to let the worker know what it needs. The second one
  // instead is filled by each worker and passed down by reference to the copier that manage any racing conditions
  // copying properly the computed data where they belong.
  struct FGGScratch {};

  // The copier structure holds the thing needed to compute the actual position on the global array
  // for the copy operation.
  struct FGGCopy
  {
    Vector<double> local_output_vector;
    // Variables needed for the Fast Gaussian evaluation.
    double diff1, diff2, diff3, ang;
    types::global_dof_index kb1, kb2, kb3, j;
    std::complex<double> cs;
    double cross, cross1;

  };

  // The worker function uses the capture to know the actual state of the BlackNUFFT class.
  // In this way we can perform the computation
  // of the column to be added at each row quite straigtforwardly. Since all the
  // workers must be able to run in parallel we must be sure that no racing condition occurs.
  auto f_fgg_worker = [this,t1,t2,t3] (IndexSet::ElementIterator j_it, FGGScratch &foo, FGGCopy &copy_data)
  {
    types::global_dof_index j=*j_it;
    // Vectors needed for the precomputations along the three dimensions. In principle they
    // belong here but in maybe in the copier they perform better.
    Vector<double> xc(2*nspread), yc(2*nspread), zc(2*nspread);

    copy_data.kb1 = types::global_dof_index(double(nf1/2) + (output_grid[0][j]-sb[0])/hs);
    copy_data.kb2 = types::global_dof_index(double(nf2/2) + (output_grid[1][j]-sb[1])/ht);
    copy_data.kb3 = types::global_dof_index(double(nf3/2) + (output_grid[2][j]-sb[2])/hu);

    copy_data.diff1 = double(nf1/2) + (output_grid[0][j]-sb[0])/hs - copy_data.kb1;
    copy_data.diff2 = double(nf2/2) + (output_grid[1][j]-sb[1])/ht - copy_data.kb2;
    copy_data.diff3 = double(nf3/2) + (output_grid[2][j]-sb[2])/hu - copy_data.kb3;

    // if(j==0)
    //   std::cout<<hu<<" "<<(output_grid[2][j]-sb[2])/hu<<" "<<nf3/2<<std::endl;
    // ang = (sk(j)-sb)*xb + (tk(j)-tb)*yb + (uk(j)-ub)*zb
    // copy_data.ang = sb[0]*input_grid[0][j] + sb[1]*input_grid[1][j] + sb[2]*input_grid[2][j];
    // copy_data.ang = (-sb[0]+output_grid[0][j])*xb[0] + (-sb[1]+output_grid[1][j])*xb[1] + (-sb[2]+output_grid[2][j])*xb[2];
    copy_data.ang = xb[0]*output_grid[0][j] + xb[1]*output_grid[1][j] + xb[2]*output_grid[2][j];
    std::complex<double> dummy1(std::cos(copy_data.ang), std::sin(copy_data.ang));
    std::complex<double> dummy2(output_vector[2*j], output_vector[2*j+1]);
    copy_data.cs = dummy1 * dummy2;
    //  if(j==3)
    //   std::cout<<diff1<<" "<<diff2<<" "<<diff3<<" "<<jb1<<" "<<jb2<<" "<<jb3<<" "<<ang<<std::endl;
    // 2) We precompute everything along x. Fast Gaussian Gridding
    // 2a) Precomptaiton in x
    // The original loop was -nspread+1 : nspread
    xc[nspread-1] = std::exp(-t1*copy_data.diff1*copy_data.diff1
                             -t2*copy_data.diff2*copy_data.diff2
                             -t3*copy_data.diff3*copy_data.diff3);

    copy_data.cross = xc[nspread-1];
    copy_data.cross1 = exp(2.*t1 * copy_data.diff1);
    for (unsigned int k1 = 0; k1 < nspread; ++k1)
      {
        copy_data.cross = copy_data.cross * copy_data.cross1;
        xc[nspread+k1] = xexp[k1]*copy_data.cross;
        // if(j==0)
        //   std::cout<<xc[nspread+k1]<<" "<<k1+1<<std::endl;
      }
    copy_data.cross = xc[nspread-1];
    copy_data.cross1 = 1./copy_data.cross1;
    for (unsigned int k1 = 0; k1 < nspread-1; ++k1) // Precomputing everything Watch out for negative indices.
      {
        copy_data.cross = copy_data.cross * copy_data.cross1;
        xc[nspread-k1-2] = xexp[k1]*copy_data.cross;
        // if(j==0)
        //   std::cout<<xc[nspread-k1-2]<<" "<<-(int)k1-1<<std::endl;

      }
    // 2b) Precomptaiton in y
    yc[nspread-1] = 1.;
    copy_data.cross = std::exp(2.*t2 * copy_data.diff2);
    copy_data.cross1 = copy_data.cross;
    for (unsigned int k2 = 0; k2 < nspread-1; ++k2) //k2 = 1, nspread-1
      {
        yc[nspread + k2] = yexp[k2]*copy_data.cross;
        yc[nspread-2-k2] = yexp[k2]/copy_data.cross;
        copy_data.cross = copy_data.cross * copy_data.cross1;
      }
    yc[2*nspread-1] = yexp[nspread-1]*copy_data.cross;
    // 2c) Precomptaiton in z
    zc[nspread-1] = 1.;
    copy_data.cross = std::exp(2.*t3 * copy_data.diff3);
    copy_data.cross1 = copy_data.cross;
    for (unsigned int k3 = 0; k3 < nspread-1; ++k3)
      {
        zc[nspread + k3] = zexp[k3]*copy_data.cross;
        zc[nspread-2-k3] = zexp[k3]/copy_data.cross;
        copy_data.cross = copy_data.cross * copy_data.cross1;
      }
    zc[2*nspread-1] = zexp[nspread-1]*copy_data.cross;
    // 2d) we must put everything together (VERY EXPENSIVE TBB)
    // We have found the nearest point, we use the gaussian gridding centered on that point
    // and we use all the precomputed stuff to compute the exponential.
    copy_data.local_output_vector.reinit(2);
    copy_data.j = j;
    for (unsigned int k3 = 0; k3<2*nspread; ++k3)
      {
        for (unsigned int k2 = 0; k2<2*nspread; ++k2)
          {
            types::global_dof_index ii = copy_data.kb1 + (copy_data.kb2+k2-(nspread-1)) * nf1 + (copy_data.kb3+k3-(nspread-1)) * nf1 * nf2;
            copy_data.cross = yc[k2] * zc[k3];
            for (unsigned int k1 = 0; k1<2*nspread; ++k1)
              {
                types::global_dof_index is2 = 2*(ii+k1-(nspread-1));
                std::complex<double> zz(fine_grid_data[is2],fine_grid_data[is2+1]);//-1 as always FORTRAN->C++
                copy_data.local_output_vector[0] +=  (xc[k1] * copy_data.cross) * zz.real();
                copy_data.local_output_vector[1] += (xc[k1] * copy_data.cross) * zz.imag();
                // if(j == 2)
                //   std::cout<<std::endl<<"second gridding "<<output_vector[2*j]<<" "<<output_vector[2*j+1]
                //   <<" "<<(int)k3-(int)(nspread-1)<<" "<<cross<<" "<<is2<<" "<<zz<<" "<<fine_grid_data[is2-1]<<std::endl;

              }
          }
      }

  };

  // The copier function uses the InitPrecCopy structure to know the global indices to add to
  // the global initial sparsity pattern. We use once again the capture to access the global memory.
  auto f_fgg_copier = [this] (const FGGCopy &copy_data)
  {
    if (fftw3_set.is_element((copy_data.kb1+copy_data.kb2*nf1+copy_data.kb3*nf1*nf2)*2))
      {

        output_vector[2*copy_data.j] += copy_data.local_output_vector[0];
        output_vector[2*copy_data.j+1] += copy_data.local_output_vector[1];
      }
  };

  // We need to create two empty structures that will be copied by WorkStream and passed
  // to each worker-copier to compute the sparsity pattern for blocks in the childlessList.
  FGGScratch foo_scratch;
  FGGCopy foo_copy;
  // output_set=complete_index_set(nk);

  WorkStream::run(output_set.begin(), output_set.end(), f_fgg_worker, f_fgg_copier, foo_scratch, foo_copy);

}

// We apply a second deconvolution to correct the first Gaussian Gridding.
// Once again this is a local operation and we can use TaskGroup inside every
// MPI processor to gain a multicore parallelism.
void BlackNUFFT::scaling_output_gridding()
{
  TimerOutput::Scope t(computing_timer, " Deconvolution after FFT ");

  double t1 = r2lamb1/(4.*numbers::PI) * hx*hx;
  double t2 = r2lamb2/(4.*numbers::PI) * hy*hy;
  double t3 = r2lamb3/(4.*numbers::PI) * hz*hz;
  if (!fft_backward)
    {
      xb[0] = -xb[0];
      xb[1] = -xb[1];
      xb[2] = -xb[2];
    }

  auto f_scaling_output_gridding = [] (IndexSet::ElementIterator j_it, double t1, double t2, double t3, std::vector<double> &output_vector_copy, const BlackNUFFT *foo_nufft)
  {
    types::global_dof_index j=*j_it;
    std::complex<double> helper(output_vector_copy[2*j],output_vector_copy[2*j+1]);
    // std::cout<<helper<<" ";
    // double foo = (std::exp(t1*(foo_nufft->output_grid[0][j]-foo_nufft->sb[0])*(foo_nufft->output_grid[0][j]-foo_nufft->sb[0])
    //                   +t2*(foo_nufft->output_grid[1][j]-foo_nufft->sb[1])*(foo_nufft->output_grid[1][j]-foo_nufft->sb[1])
    //                   +t3*(foo_nufft->output_grid[2][j]-foo_nufft->sb[2])*(foo_nufft->output_grid[2][j]-foo_nufft->sb[2])));
    helper = (std::exp(t1*(foo_nufft->output_grid[0][j]-foo_nufft->sb[0])*(foo_nufft->output_grid[0][j]-foo_nufft->sb[0])
                       +t2*(foo_nufft->output_grid[1][j]-foo_nufft->sb[1])*(foo_nufft->output_grid[1][j]-foo_nufft->sb[1])
                       +t3*(foo_nufft->output_grid[2][j]-foo_nufft->sb[2])*(foo_nufft->output_grid[2][j]-foo_nufft->sb[2])))*helper;
    double ang = (foo_nufft->output_grid[0][j]-foo_nufft->sb[0])*foo_nufft->xb[0] +
                 (foo_nufft->output_grid[1][j]-foo_nufft->sb[1])*foo_nufft->xb[1] +
                 (foo_nufft->output_grid[2][j]-foo_nufft->sb[2])*foo_nufft->xb[2];
    std::complex<double> dummy(std::cos(ang),std::sin(ang));
    helper = dummy * helper;


    output_vector_copy[2*j] = helper.real();
    output_vector_copy[2*j+1] = helper.imag();
  };

  auto f_scaling_output_gridding_tbb = [this,t1,t2,t3] (blocked_range<unsigned int> r)
  {
    for (types::global_dof_index j_it=r.begin(); j_it<r.end(); ++j_it)
      {
        types::global_dof_index j=output_set.nth_index_in_set(j_it);
        std::complex<double> helper(output_vector[2*j],output_vector[2*j+1]);
        // std::cout<<helper<<" ";
        // double foo = (std::exp(t1*(foo_nufft->output_grid[0][j]-foo_nufft->sb[0])*(foo_nufft->output_grid[0][j]-foo_nufft->sb[0])
        //                   +t2*(foo_nufft->output_grid[1][j]-foo_nufft->sb[1])*(foo_nufft->output_grid[1][j]-foo_nufft->sb[1])
        //                   +t3*(foo_nufft->output_grid[2][j]-foo_nufft->sb[2])*(foo_nufft->output_grid[2][j]-foo_nufft->sb[2])));
        helper = (std::exp(t1*(output_grid[0][j]-sb[0])*(output_grid[0][j]-sb[0])
                           +t2*(output_grid[1][j]-sb[1])*(output_grid[1][j]-sb[1])
                           +t3*(output_grid[2][j]-sb[2])*(output_grid[2][j]-sb[2])))*helper;
        double ang = (output_grid[0][j]-sb[0])*xb[0] +
                     (output_grid[1][j]-sb[1])*xb[1] +
                     (output_grid[2][j]-sb[2])*xb[2];
        std::complex<double> dummy(std::cos(ang),std::sin(ang));
        helper = dummy * helper;


        output_vector[2*j] = helper.real();
        output_vector[2*j+1] = helper.imag();
      }
  };

  Threads::TaskGroup<> scaling_output_gridding_group;
  // We need to deconvolve the output array so we use output_set.
  tbb::parallel_for(blocked_range<unsigned int> (0, output_set.n_elements(),10), f_scaling_output_gridding_tbb);
  // for (auto j_it=output_set.begin(); j_it!=output_set.end(); ++j_it)
  //   {
  //     scaling_output_gridding_group += Threads::new_task ( static_cast<void (*)(IndexSet::ElementIterator, double, double, double, std::vector<double> &, const BlackNUFFT *)> (f_scaling_output_gridding), j_it, t1, t2, t3, output_vector, this);
  //   }
  // scaling_output_gridding_group.join_all();

  // Finally we perform a AlltoAll Reduction to recover the complet
  // vector to be returned.
  Vector<double> foo(2*nk);
  MPI_Allreduce(&output_vector[0], &foo[0], 2*nk, MPI_DOUBLE, MPI_SUM, mpi_communicator);
  for (types::global_dof_index i=0; i<2*nk; ++i)
    output_vector[i] = foo[i];

}



// The run function drives the NUFFT. Essentially it needs to call all the functions in order to
// recover a proper NUFFT of type 3.
void BlackNUFFT::run()
{
  computing_timer.reset ();
  computing_timer.disable_output();
  // 1) We compute the ranges of the points.
  compute_ranges();
  // 2) We compute all tolerance related infos.
  compute_tolerance_infos();
  // 3) We create the IndexSets for the MPI parallelism.
  create_index_sets();

  input_gridding();
  // // 4) First Gaussian Gridding from the input array to
  // // the fine grid
  // fast_gaussian_gridding_on_input();
  // // 5) Deconvolution for the Second FGG (8)
  // scaling_input_gridding();
  // // 6) Compute the 3d FFT using FFTW
  compute_fft_3d();
  // 7) Local circular shifting
  shift_data_for_fftw3d();

  output_gridding();
  // // 8) Second FGG from the transformed fine grid to the
  // // output array.
  // fast_gaussian_gridding_on_output();
  // // 9) Deconvolution to correct the First FGG (4)
  // scaling_output_gridding();
  // fine_grid_data.reinit(0);
  computing_timer.print_summary ();
  computing_timer.reset ();

}


// TODO: ERASE??
void BlackNUFFT::prune_before()
{
  TimerOutput::Scope t(computing_timer, " First Pruning ");

  for (types::global_dof_index i3 = 0; i3<nf3/2-iw9; ++i3)
    for (types::global_dof_index i2 = 0; i2<nf2; ++i2)
      for (types::global_dof_index i1 = 0; i1<nf1; ++i1)
        {
          fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)]=0.;
          fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)+1]=0.;

        }
  for (types::global_dof_index i3 = nf3/2-iw9; i3<nf3/2+iw9; ++i3)
    {
      for (types::global_dof_index i2 = 0; i2<nf2/2-iw8; ++i2)
        for (types::global_dof_index i1 = 0; i1<nf1; ++i1)
          {
            fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)]=0.;
            fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)+1]=0.;

          }
      for (types::global_dof_index i2 = nf2/2-iw8; i2<nf2/2+iw8; ++i2)
        {
          for (types::global_dof_index i1 = 0; i1<nf1/2-iw7; ++i1)
            {
              fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)]=0.;
              fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)+1]=0.;

            }
          for (types::global_dof_index i1 = nf1/2+iw7; i1<nf1; ++i1)
            {
              fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)]=0.;
              fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)+1]=0.;

            }
        }
      for (types::global_dof_index i2 = nf2/2+iw8; i2<nf2; ++i2)
        for (types::global_dof_index i1 = 0; i1<nf1; ++i1)
          {
            fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)]=0.;
            fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)+1]=0.;

          }
    }
  for (types::global_dof_index i3 = nf3/2+iw9; i3<nf3; ++i3)
    for (types::global_dof_index i2 = 0; i2<nf2; ++i2)
      for (types::global_dof_index i1 = 0; i1<nf1; ++i1)
        {
          fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)]=0.;
          fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)+1]=0.;

        }

}


// This is the original Greengard pruned FFT put in C++. We have decided
// not to use it since FFTW is surely more optimised.
void BlackNUFFT::compute_stubborn_fft()
{
  TimerOutput::Scope t(computing_timer, " Stubborn FFT with FFTW 1d ");

  types::global_dof_index foo =std::max(nf3,nf2);
  foo = std::max(foo,nf1);
  Vector<double> dummy_vector(foo*2);
  // Vector<double> dummy_vector_2(foo*2);
  fftw_complex *dummy, *dummy2, *dummy3;
  // Vector<double> load_vector(std::max(nf3,nf2,nf1));
  fftw_plan p1,p2,p3;
  Vector<double> helper(dummy_vector.size());
  dummy = reinterpret_cast<fftw_complex *> (&dummy_vector[0]);
  // dummy3 = reinterpret_cast<fftw_complex *> (&dummy_vector_2[0]);

  if (fft_backward)
    {
      p2 = fftw_plan_dft_1d(nf2, dummy, dummy, FFTW_BACKWARD, FFTW_ESTIMATE);
      p3 = fftw_plan_dft_1d(nf3, dummy, dummy, FFTW_BACKWARD, FFTW_MEASURE);
    }
  else
    {
      p2 = fftw_plan_dft_1d(nf2, dummy, dummy, FFTW_FORWARD, FFTW_ESTIMATE);
      p3 = fftw_plan_dft_1d(nf3, dummy, dummy, FFTW_FORWARD, FFTW_ESTIMATE);
    }

  for (types::global_dof_index k2=0; k2<2*iw8+1; ++k2)
    {
      for (types::global_dof_index k1=0; k1<2*iw7+1; ++k1)
        {
          types::global_dof_index ii = (nf1/2+k1-(iw7)) + (nf2/2+k2-(iw8))*nf1 + (nf3/2)*nf1*nf2;
          types::global_dof_index is2 = 2 * ii;
          types::global_dof_index istart;
          dummy_vector = 0.;
          // if(k2==0 && k1==0)
          //   std::cout<<nf1/2+k1-(iw7)<<" "<<nf2/2+k2-(iw8)<<" "<<nf3/2<<std::endl;
          dummy_vector[0] = fine_grid_data[is2];
          dummy_vector[1] = fine_grid_data[is2+1];
          // std::cout<<"("<<fine_grid_data[is2]<<", "<<nf2/2+k2-(iw8)<<", "<<nf1/2+k1-(iw7)<<") ";
          // std::cout<<fine_grid_data[is2-1+1]<<" ";
          for (types::global_dof_index k3=1; k3<=iw9; ++k3)
            {
              is2 = 2*(ii+k3*nf1*nf2);
              dummy_vector[2*k3] = fine_grid_data[is2];
              dummy_vector[2*k3+1] = fine_grid_data[is2+1];

              is2 = 2*(ii-k3*nf1*nf2);
              dummy_vector[(nf3-k3)*2] = fine_grid_data[is2];
              dummy_vector[(nf3-k3)*2+1] = fine_grid_data[is2+1];
              // if((int)k2-(iw8-1) == 6)
              //   std::cout<<ii<<" "<<(nf3-k3+nf1*nf2*nf3)*2<<" "<<is2<<" "<<dummy_vector[(nf3-k3)*2]<<std::endl;

            }
          // std::cout<<" k2 = "<<(int)k2-iw8+1<<" , k1 = "<<(int)k1-iw7+1<<std::endl;
          // if((int)k2-(iw8-1) == 6 && (int)k1-(iw7-1) == 8)
          //   for(types::global_dof_index i=0; i<nf3; ++i)
          //   {
          //     std::cout<<" FFT on k3 : "<<dummy_vector[2*i]<<" "<<2*i<<std::endl;
          //     std::cout<<" FFT on k3 : "<<dummy_vector[2*i+1]<<" "<<2*i+1<<std::endl;
          //   }

          // dummy2 = reinterpret_cast<fftw_complex *> (&try_k3[0]);
          // fftw_plan p3_bis = fftw_plan_dft_1d(nf3, dummy2, dummy2, FFTW_BACKWARD, FFTW_ESTIMATE);
          fftw_execute(p3);

          // if((int)k2-(iw8-1) == 6 && (int)k1-(iw7-1) == 8)
          // {
          //   try_k3.print(std::cout);
          //   // fftw_execute(p3_bis);
          //
          //   for(auto i : dummy_vector.locally_owned_elements())
          //     std::cout<<" TRY on k3 : "<<try_k3[i]<<" "<<dummy_vector[i]<<" "<<i<<std::endl;
          // }


          for (types::global_dof_index k3 =0; k3<2*(iw9+nspread)+1; ++k3)
            {
              istart = 2*(ii+(((int)k3)-(iw9+nspread))*nf1*nf2);
              is2 = 2*(nf3/2+((int)k3)-(iw9+nspread));
              fine_grid_data[istart] = dummy_vector[is2];
              fine_grid_data[istart+1] = dummy_vector[is2+1];
              // if(std::abs(fine_grid_data[istart-1+1]-test_data_before_k2[istart-1+1])>1e-3)//6 8
              // {
              //       std::cout<<" Putting back : "<<istart<<" "<<std::abs(dummy_vector[is2+1]-test_data_before_k2[istart-1+1])<<" "
              //       <<dummy_vector[is2]<<" "<<test_data_before_k2[istart-1]<<" "<<is2<<" "<<k1-(iw7)<<" "<<k2-(iw8)<<std::endl;
              // }
              //
            }

        }
      // std::cout<<std::endl;
    }

  // for(types::global_dof_index i = 0; i<fine_grid_data.size()/2; ++i)
  //   // fine_grid_data[i]=test_data_before[i];
  //   if(std::abs(fine_grid_data[2*i]-test_data_before_k2[2*i])>1e-3)
  //   {
  //     // if(2*i==21406)
  //     std::cout<<"ERROR AFTER K3 TRANSFORM "<<fine_grid_data[2*i]<<" "<<test_data_before_k2[2*i]<<" "<<fine_grid_data[2*i+1]<<" "<<test_data_before_k2[2*i+1]<<" "<<2*i<<" "<<
  //                                             fine_grid_data[2*i]<<" "<<test_data_before[2*i]<<" "<<fine_grid_data[2*i+1]<<" "<<test_data_before[2*i+1]<<std::endl;
  //   }

  for (types::global_dof_index k3=0; k3<2*(iw9+nspread)+1; ++k3)
    {
      for (types::global_dof_index k1=0; k1<2*iw7+1; ++k1)
        {
          types::global_dof_index ii = (nf1/2+k1-(iw7)) + (nf3/2+k3-(iw9+nspread))*nf1*nf2 + (nf2/2)*nf1;
          types::global_dof_index is2 = 2 * ii;
          types::global_dof_index istart;
          dummy_vector = 0.;
          // if(k3==nspread && k1==0)
          //   std::cout<<nf1/2+k1-(iw7)<<" "<<nf3/2+k3-(iw9+nspread)<<" "<<nf2/2<<std::endl;

          dummy_vector[0] = fine_grid_data[is2];
          dummy_vector[1] = fine_grid_data[is2+1];
          for (types::global_dof_index k2=1; k2<=iw8; ++k2)
            {
              is2 = 2*(ii+k2*nf1);
              dummy_vector[2*k2] = fine_grid_data[is2];
              dummy_vector[2*k2+1] = fine_grid_data[is2+1];

              is2 = 2*(ii-k2*nf1);
              dummy_vector[2*(nf2-k2)] = fine_grid_data[is2];
              dummy_vector[2*(nf2-k2)+1] = fine_grid_data[is2+1];

            }
          fftw_execute(p2);
          for (types::global_dof_index k2 =0; k2<2*(iw8+nspread)+1; ++k2)
            {
              // istart = 2*(ii+(((int)k3-1)-(iw9+nspread-1))*nf1*nf2);
              // is2 = 2*(nf3/2+((int)k3-1)-(iw9+nspread-1));

              istart = 2*(ii+(((int)k2-1)-(iw8+nspread-1))*nf1);
              is2 = 2*(nf2/2+((int)k2-1)-(iw8+nspread-1));
              fine_grid_data[istart] = dummy_vector[is2];
              fine_grid_data[istart+1] = dummy_vector[is2+1];
              // std::cout<<" Putting back : "<<istart<<" "<<dummy_vector[is2]<<" "<<fine_grid_data[istart-1]<<" "<<is2<<" "<<k1-(iw7)<<" "<<k3-(iw9+nspread)<<" "<<k3<<" "<<ii<<std::endl;

            }
        }
    }

  // for(types::global_dof_index i = 0; i<fine_grid_data.size()/2; ++i)
  //   // fine_grid_data[i]=test_data_before[i];
  //   if(std::abs(fine_grid_data[2*i]-test_data_before_k1[2*i])>1e-3)
  //   {
  //     // if(2*i==21406)
  //     std::cout<<"ERROR AFTER K2 TRANSFORM "<<fine_grid_data[2*i]<<" "<<test_data_before_k1[2*i]<<" "<<fine_grid_data[2*i+1]<<" "<<test_data_before_k1[2*i+1]<<" "<<2*i<<" "<<
  //                                             fine_grid_data[2*i]<<" "<<test_data_before[2*i]<<" "<<fine_grid_data[2*i+1]<<" "<<test_data_before[2*i+1]<<std::endl;
  //   }

  for (types::global_dof_index k3=0; k3<2*(iw9+nspread)+1; ++k3)
    {
      for (types::global_dof_index k2=0; k2<2*(iw8+nspread)+1; ++k2)
        {
          types::global_dof_index ii = (nf3/2+k3-(iw9+nspread))*nf1*nf2 + (nf2/2+k2-(iw8+nspread))*nf1;
          dummy2 = reinterpret_cast<fftw_complex *> (&fine_grid_data[ii*2]);
          if (fft_backward)
            p1 = fftw_plan_dft_1d(nf1, dummy2, dummy2, FFTW_BACKWARD, FFTW_ESTIMATE);
          else
            p1 = fftw_plan_dft_1d(nf1, dummy2, dummy2, FFTW_FORWARD, FFTW_ESTIMATE);

          fftw_execute(p1);
          for (types::global_dof_index k1 = 1; k1<=iw7+nspread; k1=k1+2)
            {

              types::global_dof_index istart = 2*(ii+nf1/2+k1);
              types::global_dof_index is2 = 2*(ii+nf1/2-k1);
              fine_grid_data[istart] = -1.*fine_grid_data[istart];
              fine_grid_data[istart+1] = -1.*fine_grid_data[istart+1];
              fine_grid_data[is2] = -1.*fine_grid_data[is2];
              fine_grid_data[is2+1] = -1.*fine_grid_data[is2+1];

            }
        }
    }
  // for(types::global_dof_index i = 0; i<fine_grid_data.size()/2; ++i)
  //   // fine_grid_data[i]=test_data_before[i];
  //   if(std::abs(fine_grid_data[2*i]-test_data[2*i])>1e-3)
  //   {
  //     // if(2*i==21406)
  //     std::cout<<"ERROR AFTER K1 TRANSFORM "<<fine_grid_data[2*i]<<" "<<test_data[2*i]<<" "<<fine_grid_data[2*i+1]<<" "<<test_data[2*i+1]<<" "<<2*i<<" "<<
  //                                             fine_grid_data[2*i]<<" "<<test_data_before[2*i]<<" "<<fine_grid_data[2*i+1]<<" "<<test_data_before[2*i+1]<<std::endl;
  //   }
}

void BlackNUFFT::prune_after()
{
  TimerOutput::Scope t(computing_timer, " Prune After ");

  for (types::global_dof_index i3 = 0; i3<nf3/2-iw9-nspread; ++i3)
    for (types::global_dof_index i2 = 0; i2<nf2; ++i2)
      for (types::global_dof_index i1 = 0; i1<nf1; ++i1)
        {
          fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)]=0.;
          fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)+1]=0.;

        }
  for (types::global_dof_index i3 = nf3/2-iw9-nspread; i3<nf3/2+iw9+nspread; ++i3)
    {
      for (types::global_dof_index i2 = 0; i2<nf2/2-iw8-nspread; ++i2)
        for (types::global_dof_index i1 = 0; i1<nf1; ++i1)
          {
            fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)]=0.;
            fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)+1]=0.;

          }

      // for(types::global_dof_index i2 = nf2/2-iw8-nspread; i2<nf2/2+iw8+nspread; ++i2)
      // {
      //   for(types::global_dof_index i1 = 0; i1<nf1/2-iw7-nspread; ++i1)
      //   {
      //     fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)]=0.;
      //     fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)+1]=0.;
      //
      //   }
      //   for(types::global_dof_index i1 = nf1/2+iw7+nspread; i1<nf1; ++i1)
      //   {
      //     fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)]=0.;
      //     fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)+1]=0.;
      //
      //   }
      // }

      for (types::global_dof_index i2 = nf2/2+iw8+nspread; i2<nf2; ++i2)
        for (types::global_dof_index i1 = 0; i1<nf1; ++i1)
          {
            fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)]=0.;
            fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)+1]=0.;

          }
    }
  for (types::global_dof_index i3 = nf3/2+iw9+nspread; i3<nf3; ++i3)
    for (types::global_dof_index i2 = 0; i2<nf2; ++i2)
      for (types::global_dof_index i1 = 0; i1<nf1; ++i1)
        {
          fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)]=0.;
          fine_grid_data[2*(i3*nf1*nf2+i2*nf1+i1)+1]=0.;

        }

}
