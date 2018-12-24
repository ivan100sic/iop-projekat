#include "iopp.h"
#include "la.h"
#include "stopwatch.h"
#include <numeric>
// using namespace iopp;

/*
void medium_test() {
	auto ct = iopp::opencl_context();

	auto F = ct.mat(1000, 222);
	auto t = ct.vec(1000);
	auto w = ct.vec(222);
	auto alpha = ct.val(1e-6);

	// read data from stdin

	// train w

	auto FT = F.T();

	for (int i=0; i<10000; i++) {
		auto tmp = F.dot(w) - t;
		w = w - FT.dot(tmp) * alpha;
	}

	// write output

	auto w_out = w.get();

	for (float f : w_out)
		std::cout << f << ' ';
	std::cout << '\n';
}
*/

void simple_test() {

	stopwatch sw(0);

	auto ct = iopp::opencl_context();

	sw.tock();

	const int SZ = 1 << 20;

	la::vec a_data(SZ); 
	la::vec b_data(SZ);

	sw.tock();
	std::iota(a_data.begin(), a_data.end(), 1);
	std::iota(b_data.begin(), b_data.end(), 0);

	sw.tock();

	auto a = ct.vec(SZ);
	auto b = ct.vec(SZ);
	
	sw.tock();

	a.set(a_data);
	b.set(b_data);

	sw.tock();

	for (int i=0; i<4; i++) {
		auto c = a - b;
		sw.tock();
	}

	// std::cout << a.get() << '\n';
	// std::cout << b.get() << '\n';
	// std::cout << c.get() << '\n';
}

int main() {
	simple_test();
}