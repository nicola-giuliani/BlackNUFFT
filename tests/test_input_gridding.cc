#include "black_nufft.h"


void create_initial_data(types::global_dof_index &nj, types::global_dof_index &nk, std::vector<Vector<double> > &input_grid, Vector<double> &input_vector,std::vector<Vector<double> > &out_grid, double limit=100.)
{
  // types::signed_global_dof_index ms = nk/3;
  // types::signed_global_dof_index mt = nk/3;
  // types::signed_global_dof_index mu = nk/3;
  types::signed_global_dof_index n1 = (types::signed_global_dof_index)std::cbrt(nj);
  types::signed_global_dof_index n2 = (types::signed_global_dof_index)std::cbrt(nj);
  types::signed_global_dof_index n3 = (types::signed_global_dof_index)std::cbrt(nj);
  double grid_limit = limit;
  double R = 1.;
  double K = numbers::PI * grid_limit / R;
  for (types::signed_global_dof_index k3 = -n3/2; k3<(n3-1)/2; ++k3)
    {
      for (types::signed_global_dof_index k2 = -n2/2; k2<(n2-1)/2; ++k2)
        {
          for (types::signed_global_dof_index k1 = -n1/2; k1<(n1-1)/2; ++k1)
            {
              types::global_dof_index j = 1+(k1+n1/2) + (k2+n2/2)*n1 + (k3+n3/2)*n1*n2;
              input_grid[0][j-1] = R*numbers::PI*std::cos(-numbers::PI*((double)k1/n1));
              input_grid[1][j-1] = R*numbers::PI*std::cos(-numbers::PI*((double)k2/n2));
              input_grid[2][j-1] = R*numbers::PI*std::cos(-numbers::PI*((double)k3/n3));
              //  dcmplx(dsin(pi*j/n1),dcos(pi*j/n2))
              input_vector[2*(j-1)] = std::sin(numbers::PI*j/n1);
              input_vector[2*(j-1)+1] = std::cos(numbers::PI*j/n2);

            }
        }
    }
  for (types::global_dof_index k1 = 1; k1 <= nk; ++k1)
    {
      out_grid[0][k1-1] = K*(std::cos(k1*numbers::PI/nk));
      out_grid[1][k1-1] = K*(std::sin(-numbers::PI/2+k1*numbers::PI/nk));
      out_grid[2][k1-1] = K*(std::cos(k1*numbers::PI/nk));
    }



}

void test()
{

  types::global_dof_index nj=64, nk=64;
  double grid_limit=1.;
  double epsilon=1e-2;
  bool iflag=false;
  std::vector<Vector<double> > in_grid(3);
  Vector<double> in_vec;

  std::vector<Vector<double> > out_grid(3);
  Vector<double> out_vec;

  in_vec.reinit(2*nj);
  in_grid[0].reinit(nj);
  in_grid[1].reinit(nj);
  in_grid[2].reinit(nj);
  out_vec.reinit(2*nk);
  out_grid[0].reinit(nk);
  out_grid[1].reinit(nk);
  out_grid[2].reinit(nk);

  create_initial_data(nj,nk,in_grid,in_vec,out_grid,grid_limit);

  std::vector<double> in_vec_ptr(in_vec.size()),out_vec_ptr(out_vec.size());
  for (auto j : in_vec.locally_owned_elements())
    in_vec_ptr[j]=in_vec[j];
  for (auto j : out_vec.locally_owned_elements())
    out_vec_ptr[j]=out_vec[j];
  std::vector<std::vector<double> > in_grid_ptr(3, std::vector<double> (in_grid[0].size())), out_grid_ptr(3, std::vector<double> (in_grid[0].size()));
  for (unsigned int i =0; i<3; ++i)
    {
      for (auto j : in_grid[i].locally_owned_elements())
        in_grid_ptr[i][j]=in_grid[i][j];
      for (auto j : in_grid[i].locally_owned_elements())
        out_grid_ptr[i][j]=out_grid[i][j];
    }

  BlackNUFFT my_nufft(in_grid_ptr, in_vec_ptr, out_grid_ptr, out_vec_ptr);
  my_nufft.init_nufft(epsilon, iflag);
  my_nufft.compute_ranges();
  my_nufft.compute_tolerance_infos();
  my_nufft.create_index_sets();
  my_nufft.input_gridding();
  my_nufft.computing_timer.disable_output();
  my_nufft.compute_fft_3d();
  my_nufft.shift_data_after_fft();
  my_nufft.output_gridding();
  my_nufft.fine_grid_data.locally_owned_elements().print(std::cout);
  for (auto i : my_nufft.fine_grid_data.locally_owned_elements())
    std::cout<<my_nufft.fine_grid_data[i]<<" ";
  std::cout<<std::endl;


}

int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  using namespace tbb;

  test();


  return 0;
}
