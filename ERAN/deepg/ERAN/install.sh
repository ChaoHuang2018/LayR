#!/bin/bash

wget ftp://ftp.gnu.org/gnu/m4/m4-1.4.1.tar.gz
tar -xvzf m4-1.4.1.tar.gz
cd m4-1.4.1
./configure
make
make install
cp src/m4 /usr/bin
cd ..
rm m4-1.4.1.tar.gz



wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
tar -xvf gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --enable-cxx
make
make install
cd ..
rm gmp-6.1.2.tar.xz



wget https://www.mpfr.org/mpfr-current/mpfr-4.0.2.tar.xz
tar -xvf mpfr-4.0.2.tar.xz
cd mpfr-4.0.2
./configure
make
make install
cd ..
rm mpfr-4.0.2.tar.xz



git clone https://github.com/eth-sri/ELINA.git
cd ELINA
git checkout fe565031
./configure
make
make install
cd ..

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/usr/lib

ldconfig



