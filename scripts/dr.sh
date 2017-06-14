docker run -u root -P -v  `pwd`:/home/dealii/app:rw -t   limmerkate/deal-fftw  /bin/sh -c "$@"

