#FROM mathlab/deal2lkit:latest
FROM dealii/dealii:9.0.0-gcc-mpi-fulldepsspack-debugrelease

USER root

#fftw

RUN cd $HOME 
ENV SPACK_ROOT $HOME/spack &&\ 
mkdir SPACK_ROOT &&\ 
cd $SPACK_ROOT &&\ 
git clone https://github.com/spack/spack.git . &&\ 
git checkout develop &&\ 
git reset --hard 99f678035a279cf85a32a4fdbd98a117b9c1832f &&\ 
spack install environment-modules &&\ 
source $SPACK_ROOT/share/spack/setup-env.sh &&\ 
spack install pfft &&\ 
spack load pfft 

#ENV FFTW_DIR HOME/libs/fftw-3.3.6-pl2 &&\
#ENV PFFT_DIR HOME/libs/fftw-3.3.6-pl2 

#RUN wget http://www.fftw.org/fftw-3.3.6-pl2.tar.gz &&\
#    tar xf fftw-3.3.6-pl2.tar.gz && rm -f fftw-3.3.6-pl2.tar.gz && \
#    cd fftw-3.3.6-pl2 && \
#    mkdir build && \
#    ./configure --enable-threads --enable-mpi --prefix=/$HOME/libs/fftw-3.3.6-pl2/ --#enable-shared && \
#    make install && \
#    cd $HOME && \
#    rm -rf fftw-3.3.6-pl2
#ENV FFTW_DIR $HOME/libs/fftw-3.3.6-pl2

USER dealii