//-----------------------------------------------------------
//
//    Copyright (C) 2017 by Nicola Giuliani
//
//    This file is subject to GPL and may not be distributed
//    without copyright and license information. Please refer
//    to the file BlackNUFFT/LICENSE for the  text  and
//    further information on this license.
//
//-----------------------------------------------------------



#include <deal.II/lac/vector.h>
#include <deal.II/lac/parallel_vector.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/base/types.h>
#include <deal.II/base/timer.h>

#include <pfft.h>


#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>

// A typedef for a the signed integer of size global_dof_index
DEAL_II_NAMESPACE_OPEN
namespace types
{
#ifdef DEAL_II_WITH_64BIT_INDICES
  typedef long long int signed_global_dof_index;
#else
  typedef int signed_global_dof_index;
#endif
}
DEAL_II_NAMESPACE_CLOSE

using namespace dealii;

/** The class that implements the Non Uniform FFT of type 3.
*/
class BlackNUFFT
{
public:
  friend void test();
  /** Class constructor, we need the input - output grids and the relative vectors. We require
  an MPI communicator to set up the distributed fine grid vector. By default we consider MPI_COMM_WORLD*/
  BlackNUFFT(const std::vector<std::vector<double> > &in_grid, const std::vector<double> &in , std::vector<std::vector<double> > &out_grid, std::vector<double> &out, MPI_Comm comm_in=MPI_COMM_WORLD);

  /** Class constructor, we need the input - output grids and the relative vectors. We require
  an MPI communicator to set up the distributed fine grid vector. By default we consider MPI_COMM_WORLD*/
  void init_nufft(double eps, bool fft_bool, unsigned int tbb_granularity_in=10, std::string gridding_input="FGG", std::string fft_input="FFTW");

  /** The driver of the function. It calls all the needed function of the private part*/
  void run();

  // void read_grid(std::string filename);
  //
  // void compute_error(std::string filename);

  // // Just a bunch of test functions we needed to perform debugging.
  // Vector<double> test_data;
  // Vector<double> test_data_before;
  // Vector<double> test_data_before_k2;
  // Vector<double> test_data_before_k1;
  // Vector<double> true_solution;
  // Vector<double> try_k3;
  // Vector<double> final_helper;

private:

  /** It simply computes the bounding box of both the input and output arrays.*/
  void compute_ranges();

  /** Once the bounding box is computed we can use epsilon (aka the tolerance)
  to compute the fine grid box and the convolution - deconvolution constants.*/
  void compute_tolerance_infos();

  /** The function that sets up all the IndexSets needed by the MPI parallelisation.
  We need one for the output grid, this must not have more than 2^32 elements (50GB of memory) for the
  inner characterisitcs of the distributed vectors we use. We divide it following the requirements
  of FFT 3d provided by FFTW. Then we need to split the input and output
  vector to subdivide the gridding and deconvolution cycles. */
  void create_index_sets();

  /** A further index creator. In this way we split the input grid in slices along the
  second variable to obtain a higher scalability in terms of shared memory per each MPI
  processor.*/
  void create_index_sets_for_first_gridding(const unsigned int sets_number = 2);

  /** This functions computes the gridding  from the input vector to
  the distributed fine grid array. It allows for different gridding choices.*/
  void input_gridding();
  /** This functions computes the Fast Gaussian Gridding from the input vector to
  the distributed fine grid array. This has been parallelised using two nested TBB
  WorkStream functions to achieve maximum shared memory parallelism, together with a
  MPI work balance.*/
  void fast_gaussian_gridding_on_input();

  /** This function computes the deconvolution of the second FGG, done by fast_gaussian_gridding_on_output.
  We perform the gridding on the fine grid data. It is a puntual operation on each processor regarding
   the owned fine data array.*/
  void scaling_input_gridding();

  /** This function calls the MPI fft3d using the FFTW or PFFT package.*/
  void compute_fft_3d();

  void prepare_pfft_array(pfft_complex *in);

  void retrieve_pfft_result(pfft_complex *out);

  /** This function performs a circular shift on the transformed array to obtain an overall shifted FFT.
  It is a local multiplication of -1.*/
  void shift_data_after_fft();
  void shift_data_before_fft();

  /** This functions computes the gridding  from the distributed fine grid array  to
  the output vector. It allows for different gridding choices.*/
  void output_gridding();

  /** This functions computes the Fast Gaussian Gridding from the fine array to
  the output vector. This has been parallelised using TBB
  WorkStream functions to achieve maximum shared memory parallelism. We only computed the
  values owned by the processor so no racing conditions occur.*/
  void fast_gaussian_gridding_on_output();

  /* Second deconvolution on the output array to correct the effect of fast_gaussian_gridding_on_input();*/
  void scaling_output_gridding();


  void prune_before();

  void prune_after();

  void compute_stubborn_fft();

  /// Reference to the input grid.
  const std::vector<std::vector<double> > &input_grid;
  /// Reference to the output_grid grid.
  std::vector<std::vector<double> > &output_grid;
  /// Reference to the input ventor.
  const std::vector<double> &input_vector;
  /// Reference to the output ventor.
  std::vector<double> &output_vector;

  /// The dimensions of the input box.
  std::vector<double> xb;
  /// The center of the input box.
  std::vector<double> xm;
  /// The dimensions of the output box.
  std::vector<double> sb;
  /// The center of the output box.
  std::vector<double> sm;

  /// Overall number of input points
  types::global_dof_index nj;

  /// Overall number of output points
  types::global_dof_index nk;

  /// Prescribed tolerance
  double epsilon;

  /// String parameters specifying the kind of gridding and FFT
  std::string gridding, fft_type;

  // Bunch of precision realted parameters

  double rat, rat1, rat2, rat3;

  types::global_dof_index nf1, nf2, nf3;

  ptrdiff_t input_offset[3], output_offset[3];
  ptrdiff_t local_i_start_shift[3], local_i_start[3], local_ni[3];
  ptrdiff_t local_o_start_shift[3], local_o_start[3], local_no[3];
  ptrdiff_t local_n[3], ni[3], no[3], complete_n[3];
  std::vector<unsigned int> iblock, oblock;

  types::global_dof_index local_nf3, local_nf3_start;

  double r2lamb1, r2lamb2, r2lamb3;

  types::global_dof_index iw7, iw8, iw9;

  Vector<double> deconv_array_x, deconv_array_y, deconv_array_z;

  IndexSet input_set, output_set, fftw3_set, fftw3_output_set, pfft_input_set, pfft_output_set, fft_input_set, fft_output_set;

  std::vector<std::vector<IndexSet> > grid_sets;

  parallel::distributed::Vector<double> fine_grid_data, grid_data_input, grid_data_output;
  parallel::distributed::Vector<double> *input_grid_helper;

  /// Granularity for tbb parallel fors
  unsigned int tbb_granularity;

  /// The spread of the gaussian convolution
  unsigned int nspread;

  // The increment on the fine grids (x,y,z origin space), (s,t,u reciprocal space)
  double hx, hy, hz;

  double hs, ht, hu;

  /// Variable to determine the sense of FFT.
  bool fft_backward;

  /// Three vectors to store the precomputation of the exponentials.
  Vector<double> xexp, yexp, zexp;

  MPI_Comm mpi_communicator, comm_cart_2d;

  ptrdiff_t alloc_local_pfft;

  unsigned int n_mpi_processes, this_mpi_process;

  ConditionalOStream pcout;


public:
  TimerOutput  computing_timer;
  // void
};
