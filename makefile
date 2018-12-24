test: test.cpp iopp.cpp iopp.h kernels.c stopwatch.h la.h
	g++ -O2 -Wall -lOpenCL test.cpp iopp.cpp -o test