#include "nufft_3d_3.h"
// #include "fftw3.h"
#include <iostream>
#include <fstream>


void read_grid(std::vector<Vector<double> > &input_grid, Vector<double> &input_vector, std::string filename)
{
  std::ifstream infile (filename.c_str());
  std::string instring;
  types::global_dof_index i;
  i=0;
  while(infile.good() && i < input_grid[0].size())
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
  while(infile.good() && i < input_vector.size())
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
  for(k=0; k<fk1.size()/2; ++k)
  {
     ealg += (fk1(k*2)-fk0(k*2))*(fk1(k*2)-fk0(k*2))+(fk1(k*2+1)-fk0(k*2+1))*(fk1(k*2+1)-fk0(k*2+1));
     std::complex<double> foo(fk1(k*2),fk1(k*2+1));
     salg += std::abs(foo)*std::abs(foo);
    //  std::cout<<ealg<<" "<<salg<<" "<<k<<std::endl;
  }
  err = std::sqrt(ealg/salg);
  return err;

}

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
  for(types::signed_global_dof_index k3 = -n3/2; k3<(n3-1)/2; ++k3)
  {
     for(types::signed_global_dof_index k2 = -n2/2; k2<(n2-1)/2; ++k2)
     {
        for(types::signed_global_dof_index k1 = -n1/2; k1<(n1-1)/2; ++k1)
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
    for(types::global_dof_index k1 = 1; k1 <= nk; ++k1)
    {
       out_grid[0][k1-1] = K*(std::cos(k1*numbers::PI/nk));
       out_grid[1][k1-1] = K*(std::sin(-numbers::PI/2+k1*numbers::PI/nk));
       out_grid[2][k1-1] = K*(std::cos(k1*numbers::PI/nk));
    }



}


int main(int argc, char *argv[])
{
  unsigned int threads=numbers::invalid_unsigned_int;
  unsigned int check_results=0;
  if(argc > 2)
    threads = atoi(argv[2]);
  if(argc > 1)
    check_results = atoi(argv[1]);
  // argc=NULL;
  // argv=NULL;
  if(threads == 0)
    threads=numbers::invalid_unsigned_int;
  std::string input_grid_x, input_grid_y, input_grid_z;
  std::string output_grid_x, output_grid_y, output_grid_z;
  std::string input_vector_file, output_vector_file;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, threads);
  types::global_dof_index nj = 864000;//3000000;//864000
  types::global_dof_index nk = 864000;//3000000;//864
  double epsilon = 1e-5;//9.9999999999999998E-017;
  double grid_limit = 100.;
  bool iflag = false;


  std::vector<Vector<double> > in_grid(3);
  Vector<double> in_vec;

  std::vector<Vector<double> > out_grid(3);
  Vector<double> out_vec;
  Vector<double> out_vec1;//(nk*2);
  // Vector<double> out_vec2(nk*2);

  if(argc == 12)
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

  if(check_results==1)
  {
    read_grid(in_grid,in_vec,"../origin.txt");

    if(iflag)
      read_grid(out_grid,out_vec1,"../results_backward.txt");
    else
      read_grid(out_grid,out_vec1,"../results_forward.txt");
  }
  else if(check_results==0)
    create_initial_data(nj,nk,in_grid,in_vec,out_grid,grid_limit);


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
  for(auto j : in_vec.locally_owned_elements())
    in_vec_ptr[j]=in_vec[j];
  for(auto j : out_vec.locally_owned_elements())
    out_vec_ptr[j]=out_vec[j];

  std::vector<std::vector<double> > in_grid_ptr(3, std::vector<double> (in_grid[0].size())), out_grid_ptr(3, std::vector<double> (in_grid[0].size()));
  for(unsigned int i =0; i<3; ++i)
  {
    for(auto j : in_grid[i].locally_owned_elements())
      in_grid_ptr[i][j]=in_grid[i][j];
    for(auto j : in_grid[i].locally_owned_elements())
      out_grid_ptr[i][j]=out_grid[i][j];
  }
  std::cout<<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)<<" "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
  NUFFT3D3 my_nufft(in_grid_ptr, in_vec_ptr, out_grid_ptr, out_vec_ptr);
  my_nufft.init_nufft(epsilon, iflag);

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

  if(check_results == 2 && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
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
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0 && check_results == 1)
  {
    double error = errcomp(out_vec, out_vec1, out_grid);
    std::cout<<"error from Greengard's version"<<std::endl;
    std::cout<<error<<" "<<epsilon<<std::endl;
  }
  // std::cout <<"Hello World"<<std::endl;
}
