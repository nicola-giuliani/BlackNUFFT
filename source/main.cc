#include "black_nufft.h"
// #include "fftw3.h"
#include <iostream>
#include <fstream>
#include <pfft.h>

void read_grid(std::vector<Vector<double> > &input_grid, Vector<double> &input_vector, std::string filename)
{
  std::ifstream infile (filename.c_str());
  std::string instring;
  types::global_dof_index i;
  i=0;
  while (infile.good() && i < input_grid[0].size())
    {
      getline ( infile, instring, '\n');
      // std::ifstream ifs1(instring);
      // ifs1 >> input_grid[0][i];
      // std::cout<< input_grid[0][i];
      input_grid[0][i]=double(atof(instring.c_str()));
      instring.clear();
      getline ( infile, instring, '\n');
      // std::ifstream ifs2(instring);
      // ifs2 >> input_grid[1][i];
      // std::cout<< input_grid[1][i];
      input_grid[1][i]=double(atof(instring.c_str()));
      instring.clear();

      //
      // getline ( infile, instring, ' ');
      // getline ( infile, instring, ' ');
      // getline ( infile, instring, ' ');
      // instring.clear();
      // getline ( infile, instring, ' ');
      // input_grid[1][i]=atof(instring.c_str());
      // instring.clear();
      getline ( infile, instring, '\n');
      //
      // getline ( infile, instring, ' ');
      // getline ( infile, instring, ' ');
      // getline ( infile, instring, ' ');
      // instring.clear();
      // getline ( infile, instring, ' ');
      input_grid[2][i]=atof(instring.c_str());
      instring.clear();
      getline ( infile, instring, '(');
      instring.clear();
      getline ( infile, instring, ',');
      input_vector[2*i]=atof(instring.c_str());
      instring.clear();
      getline ( infile, instring, ')');
      input_vector[2*i+1]=atof(instring.c_str());
      instring.clear();
      getline ( infile, instring, '\n');
      instring.clear();
      // std::cout<<input_grid[0][i]<<" "<<input_grid[1][i]<<" "<<input_grid[2][i]<<" "<<std::endl;
      i = i+1;


    }

}

void read_nufft_data(Vector<double> &input_vector, std::string filename)
{
  std::ifstream infile (filename.c_str());
  std::string instring;
  types::global_dof_index i;
  i=0;
  while (infile.good() && i < input_vector.size())
    {
      instring.clear();
      getline ( infile, instring, '\n');
      input_vector[i] = double(atof(instring.c_str()));
      // if(input_vector[i]!=0)
      //   std::cout<<instring<<" "<<input_vector[i]<<" "<<double(atof(instring.c_str()))<<std::endl;
      i=i+1;
    }

}

double errcomp(Vector<double> &fk0, Vector<double> &fk1,std::vector<Vector<double> > output_grid)
{
  Assert(fk0.size() == fk1.size(), ExcMessage("Uncompatible vectors!"));
  unsigned int k;
  double salg,ealg,err;
  ealg = 0.;
  salg = 0.;

  // for(unsigned int i=0; i<fk0.size(); ++i)
  // {
  //   std::cout<<i/2<<" "<<output_grid[0].size()<<" ";
  //   std::cout<<fk0[i]<<" "<<fk1[i]<<" "<<output_grid[0][i/2]<<" "<<output_grid[1][i/2]<<" "<<output_grid[2][i/2]<<std::endl;
  // }
  // fk0.print(std::cout);
  // fk1.print(std::cout);
  // for(unsigned int i=0; i<fk0.size(); ++i)
  //   foo[i]=std::abs(fk1[i]-fk0[i]);
  // foo.print(std::cout);
  for (k=0; k<fk1.size()/2; ++k)
    {
      ealg += (fk1(k*2)-fk0(k*2))*(fk1(k*2)-fk0(k*2))+(fk1(k*2+1)-fk0(k*2+1))*(fk1(k*2+1)-fk0(k*2+1));
      std::complex<double> foo(fk1(k*2),fk1(k*2+1));
      salg += std::abs(foo)*std::abs(foo);
       // std::cout<<ealg<<" "<<salg<<" "<<k<<std::endl;
    }
  err = std::sqrt(ealg/salg);
  return err;

}

void create_initial_data(types::global_dof_index &nj, types::global_dof_index &nk, std::vector<Vector<double> > &input_grid, Vector<double> &input_vector,std::vector<Vector<double> > &out_grid, double limit=100.)
{
  // types::signed_global_dof_index ms = nk/3;
  // types::signed_global_dof_index mt = nk/3;
  // types::signed_global_dof_index mu = nk/3;
  std::cout<<"Creating initial data"<<std::endl;
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


void my_function_pfft()
{
  ptrdiff_t np[2], complete_n[3], ni[3], no[3], local_ni[3], local_no[3], local_i_start[3], local_o_start[3], alloc_local;
  int size;
  MPI_Comm comm_cart_2d;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  np[0] = size;
  np[1] = 1;

  ni[0] = 200; ni[1] = 200; ni[2] = 200;
  no[0] = 100; no[1] = 100; no[2] = 100;

  complete_n[0] = 5000; complete_n[1] = 500; complete_n[2] = 500;

  ptrdiff_t howmany = 1;

  std::cout<<" CREATING PROC MESH "<<std::endl;
  pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &comm_cart_2d);


  std::cout<<" COMPUTING LOCAL SIZES "<<std::endl;
  alloc_local = pfft_local_size_many_dft(3, complete_n, ni, no, howmany,
      PFFT_DEFAULT_BLOCKS, PFFT_DEFAULT_BLOCKS,
      comm_cart_2d, PFFT_TRANSPOSED_NONE,
      local_ni, local_i_start, local_no, local_o_start);

  std::cout<<Utilities::MPI::this_mpi_process(comm_cart_2d)<<" "<<local_ni[0]*local_ni[1]*local_ni[2]<<" "<<alloc_local<<std::endl;
  std::cout<<Utilities::MPI::this_mpi_process(comm_cart_2d)<<" "<<local_no[0]*local_no[1]*local_no[2]<<" "<<alloc_local<<std::endl;
  std::cout<<Utilities::MPI::this_mpi_process(comm_cart_2d)<<" "<<complete_n[0]*complete_n[1]*complete_n[2]<<" "<<alloc_local<<std::endl;
  pfft_complex * in;
  in = new pfft_complex[alloc_local];//[local_ni[0]*local_ni[1]*local_ni[2]];
  pfft_complex * out;
  out = new pfft_complex[alloc_local];//[local_no[0]*local_no[1]*local_no[2]];

  std::cout<<" PLAN "<<std::endl;
  pfft_plan pfft_plan = pfft_plan_many_dft(
    3, complete_n, ni, no, howmany, PFFT_DEFAULT_BLOCKS, PFFT_DEFAULT_BLOCKS,
    in, out, comm_cart_2d, PFFT_BACKWARD, PFFT_TRANSPOSED_NONE | PFFT_ESTIMATE);//PFFT_TRANSPOSED_NONE| PFFT_MEASURE| PFFT_DESTROY_INPUT);

    ptrdiff_t m = 0;
    for(ptrdiff_t k0=0; k0 < local_ni[0]; k0++)
      for(ptrdiff_t k1=0; k1 < local_ni[1]; k1++)
        for(ptrdiff_t k2=0; k2 < local_ni[2]; k2++)
        {
          std::cout<<in[m][0]<<" + j "<<in[m][1]<<" , ";
          m = m+1;
        }

 std::cout<<std::endl<<" EXECUTE "<<std::endl;
 pfft_execute(pfft_plan);
 std::cout<<" FINISH "<<std::endl;
  m = 0;
 for(ptrdiff_t k0=0; k0 < local_no[0]; k0++)
   for(ptrdiff_t k1=0; k1 < local_no[1]; k1++)
     for(ptrdiff_t k2=0; k2 < local_no[2]; k2++)
     {
       std::cout<<out[m][0]<<" + j "<<out[m][1]<<" , ";
       m = m+1;

     }


 pfft_destroy_plan(pfft_plan);

}
int main(int argc, char *argv[])
{

  unsigned int threads=numbers::invalid_unsigned_int;
  unsigned int check_results=0;
  if (argc > 2)
    threads = atoi(argv[2]);
  if (argc > 1)
    check_results = atoi(argv[1]);
  // argc=NULL;
  // argv=NULL;
  if (threads == 0)
    threads=numbers::invalid_unsigned_int;
  std::string input_grid_x, input_grid_y, input_grid_z;
  std::string output_grid_x, output_grid_y, output_grid_z;
  std::string input_vector_file, output_vector_file;

  std::cout<<threads<<std::endl;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, threads);

  types::global_dof_index nj = 84;//8;//6;//4;//3000000;//864000
  types::global_dof_index nk = 84;//8;//64;//3000000;//864
  double epsilon = 1e-5;//9.9999999999999998E-017;
  double grid_limit = 1.;
  bool iflag = false;

  // my_function_pfft();
  // return 1;

  std::vector<Vector<double> > in_grid(3);
  Vector<double> in_vec;

  std::vector<Vector<double> > out_grid(3);
  Vector<double> out_vec;
  Vector<double> out_vec1;//(nk*2);
  // Vector<double> out_vec2(nk*2);

  if (argc == 12)
    {
      input_grid_x = argv[3];
      input_grid_y = argv[4];
      input_grid_z = argv[5];
      output_grid_x = argv[6];
      output_grid_y = argv[7];
      output_grid_z = argv[8];
      input_vector_file = argv[9];
      output_vector_file = argv[10];
      iflag = atoi(argv[11]);
      std::ifstream in_vecty(input_vector_file.c_str());
      in_vec.block_read(in_vecty);
      nj = in_vec.size()/2;
      std::ifstream in_griddy_x(input_grid_x.c_str());
      in_grid[0].block_read(in_griddy_x);
      std::ifstream in_griddy_y(input_grid_y.c_str());
      in_grid[1].block_read(in_griddy_y);
      std::ifstream in_griddy_z(input_grid_z.c_str());
      in_grid[2].block_read(in_griddy_z);
      // std::ifstream out_vecty(output_vector_file.c_str());
      // out_vec.block_read(out_vecty);
      std::ifstream out_griddy_x(output_grid_x.c_str());
      out_grid[0].block_read(out_griddy_x);
      std::ifstream out_griddy_y(output_grid_y.c_str());
      out_grid[1].block_read(out_griddy_y);
      std::ifstream out_griddy_z(output_grid_z.c_str());
      out_grid[2].block_read(out_griddy_z);
      nk=out_grid[0].size();
      out_vec.reinit(2*nk);

    }
  else
    {
      in_vec.reinit(2*nj);
      in_grid[0].reinit(nj);
      in_grid[1].reinit(nj);
      in_grid[2].reinit(nj);
      out_vec.reinit(2*nk);
      out_vec1.reinit(2*nk);
      out_grid[0].reinit(nk);
      out_grid[1].reinit(nk);
      out_grid[2].reinit(nk);
    }

  std::cout<<check_results<<std::endl;
  if (check_results==1)
    {
      read_grid(in_grid,in_vec,"../origin.txt");

      if (iflag)
        read_grid(out_grid,out_vec1,"../results_backward.txt");
      else
        read_grid(out_grid,out_vec1,"../results_forward.txt");
    }
  else if (check_results==0)
    create_initial_data(nj,nk,in_grid,in_vec,out_grid,grid_limit);


  // for(unsigned int i=0; i<3; ++i)
  //   // for(auto j : out_grid[i].locally_owned_elements())
  //     out_grid[i].sadd(0.,numbers::PI * grid_limit ,in_grid[i]);
  double foo;
  // std::vector<VectorView<double> > in_foo(3,VectorView<double>(1,&foo));
  // std::vector<VectorView<double> > out_foo(3,VectorView<double>(1,&foo));
  // for(unsigned int i=0; i<3; ++i)
  // {
  //   in_foo[i].reinit(nj, &in_grid[i][0]);
  //   out_foo[i].reinit(nk, &out_grid[i][0]);
  // }
  // A trick to increase quickly the fine grid
  // out_grid[0] *= 2.;
  // out_grid[1] *= 2.;
  // out_grid[2] *= 2.;
  // Still buggy
  // create_initial_data(nj,nk,in_grid,in_vec,out_grid);
  // VectorView<double> foo_in(nj*2, &in_vec[0]);
  // VectorView<double> foo_out(nk*2, &out_vec[0]);
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
  std::cout<<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)<<" "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
  BlackNUFFT my_nufft(in_grid_ptr, in_vec_ptr, out_grid_ptr, out_vec_ptr);
  my_nufft.init_nufft(epsilon, iflag, 10,"FGG","PFFT");

  // my_nufft.test_data.reinit(40*40*40*2);//test for eps=1e-4
  // my_nufft.test_data_before.reinit(80*80*80*2);//test for eps=1e-4
  // my_nufft.test_data_before_k2.reinit(40*40*40*2);
  // my_nufft.test_data_before_k1.reinit(40*40*40*2);
  // my_nufft.final_helper.reinit(40*40*40*2);
  // read_nufft_data(my_nufft.test_data_before,"../test_before.txt");
  // read_nufft_data(my_nufft.test_data_before_k2,"../test_before_k2.txt");
  // read_nufft_data(my_nufft.test_data_before_k1,"../test_before_k1.txt");
  // read_nufft_data(my_nufft.final_helper,"../final_test.txt");
  // my_nufft.try_k3.reinit(60);
  // read_nufft_data(my_nufft.test_data,"../test.txt");
  // read_nufft_data(out_vec2,"../final_test.txt");
  // read_nufft_data(my_nufft.try_k3,"../test_k3.txt");
  // // test.print(std::cout);
  // my_nufft.true_solution.reinit(nk*2);
  // my_nufft.true_solution = out_vec2;
  // for(types::global_dof_index i=0; i<test.size(); ++i)
  // {
  //   my_nufft.test_data[i] = test[i];
  // }
  // my_nufft.test_data->print(std::cout);
  my_nufft.run();
  for (auto j : out_vec.locally_owned_elements())
    out_vec[j]=out_vec_ptr[j];

  // out_vec.print(std::cout);
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{
  input_vector_file = "try_fftw.txt";
  std::ifstream in_vecty(input_vector_file.c_str());
  out_vec1.block_read(in_vecty);
  std::cout<<out_vec.size()<<out_vec1.size()<<std::endl;
  double error = errcomp(out_vec, out_vec1, out_grid);
  std::cout<<"error from FFTW version"<<std::endl;
  std::cout<<error<<" "<<epsilon<<std::endl;

  for (auto j : out_vec.locally_owned_elements())
      if(std::abs(out_vec[j]-out_vec1[j])>1e-13)
        std::cout<<out_vec[j]<<" "<<out_vec1[j]<<std::endl;
    }

  // output_vector_file = "try_fftw.txt";
  // std::ofstream output_veccy(output_vector_file.c_str());
  // out_vec.block_write(output_veccy);

  if (check_results == 2 && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    {
      // std::string input_grid_x="input_grid_x.bin";
      // std::string input_grid_y="input_grid_y.bin";
      // std::string input_grid_z="input_grid_z.bin";
      // std::string input_vector_file="input_file.bin";
      // std::string output_grid_x="output_grid_x.bin";
      // std::string output_grid_y="output_grid_y.bin";
      // std::string output_grid_z="output_grid_z.bin";
      // std::ofstream in_griddy_x(input_grid_x.c_str());
      // std::ofstream in_griddy_y(input_grid_y.c_str());
      // std::ofstream in_griddy_z(input_grid_z.c_str());
      // std::ofstream input_veccy(input_vector_file.c_str());
      // std::ofstream out_griddy_x(output_grid_x.c_str());
      // std::ofstream out_griddy_y(output_grid_y.c_str());
      // std::ofstream out_griddy_z(output_grid_z.c_str());
      // in_grid[0].block_write(in_griddy_x);
      // in_grid[1].block_write(in_griddy_y);
      // in_grid[2].block_write(in_griddy_z);
      // in_vec.block_write(input_veccy);
      // out_grid[0].block_write(out_griddy_x);
      // out_grid[1].block_write(out_griddy_y);
      // out_grid[2].block_write(out_griddy_z);
      // std::string output_vector_file="output_file.bin";
      std::ofstream output_veccy(output_vector_file.c_str());
      out_vec.block_write(output_veccy);

      // file_name1 = "stokes_forces_" + Utilities::int_to_string(cycle) + ".bin";
      // std::ofstream forces (file_name1.c_str());
      // loc_stokes_forces.block_write(forces);

    }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0 && check_results == 1)
    {

      double error = errcomp(out_vec, out_vec1, out_grid);
      std::cout<<"error from Greengard's version"<<std::endl;
      std::cout<<error<<" "<<epsilon<<std::endl;
    }
  // std::cout <<"Hello World"<<std::endl;
}
