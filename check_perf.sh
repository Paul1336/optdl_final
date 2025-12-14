#!/bin/bash

# Single core for stable numbers
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=$HOME/OpenBLAS/installation/lib:$LD_LIBRARY_PATH

for atn in "exp1_naive" "exp4_flash"
do
    fname="test_$atn"
    echo $fname
    echo "O2"

    rm -rf $fname
    g++ -std=c++20 -O2 -pedantic -Wall -Wextra -march=native \
        $atn.cpp \
        -o "./$fname"


    taskset -c 5 "./$fname" > timing_O2_$atn.log
    echo ""

    fname="test_$atn"
    echo $fname
    echo "O3"

    rm -rf $fname
    g++ -std=c++20 -O3 -pedantic -Wall -Wextra -march=native \
        $atn.cpp \
        -o "./$fname"


    taskset -c 5 "./$fname" > timing_O3_$atn.log
    echo ""

    fname="test_$atn"
    echo $fname
    echo "O2_ASAN"

    rm -rf $fname
    g++ -fsanitize=address -std=c++20 -O2 -pedantic -Wall -Wextra -march=native \
        $atn.cpp \
        -o "./$fname"


    taskset -c 5 "./$fname" > timing_ASAN_$atn.log
    echo ""
done

for atn in "exp2_falsh_1" "exp2_falsh_2" "exp5_flash_IO" "exp5_flash_math" "exp5_naive_IO" "exp5_naive_math"
do
    fname="test_$atn"
    echo $fname
    echo "O2"

    rm -rf $fname
    g++ -std=c++20 -O2 -pedantic -Wall -Wextra -march=native \
        $atn.cpp \
        -o "./$fname"


    taskset -c 5 "./$fname" > timing_O2_$atn.log
    echo ""
done