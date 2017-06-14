FROM mathlab/deal2lkit:latest


USER root

#fftw

RUN wget http://www.fftw.org/fftw-3.3.6-pl2.tar.gz &&\
    tar xf fftw-3.3.6-pl2.tar.gz && rm -f fftw-3.3.6-pl2.tar.gz && \
    cd fftw-3.3.6-pl2 && \
    mkdir build && \
    ./configure --enable-mpi --prefix=/$HOME/libs/fftw-3.3.6-pl2/ --enable-shared && \
    make install && \
    cd $HOME && \
    rm -rf fftw-3.3.6-pl2
ENV FFTW_DIR $HOME/libs/fftw-3.3.6-pl2

#USER dealii