test: test.cpp iopp.cpp iopp.h kernels.c stopwatch.h la.h makefile
	g++ -std=c++14 -O2 -Wall test.cpp iopp.cpp -o test -lOpenCL

mnist: mnist.cpp iopp.cpp iopp.h kernels.c stopwatch.h la.h makefile
	g++ -std=c++14 -O2 -Wall mnist.cpp iopp.cpp -o mnist -lOpenCL