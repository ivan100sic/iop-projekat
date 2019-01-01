#include "iopp.h"
#include "la.h"
#include "stopwatch.h"
#include <numeric>
#include <algorithm>
// using namespace iopp;

auto ct = iopp::opencl_context();

void compile_check() {
	auto v = ct.vec(10);
	v.set({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
	v = v - v * v / v + v;
	v.get();
	v += v -= v *= v /= v;

	auto a = ct.val(3.33);
	v = (v + a - a) * a / a;
	v += v -= v *= v /= a;
}

void medium_test() {
	int n = 8192, m = 8192;

	auto F = ct.mat(n, m);
	auto t = ct.vec(n);
	auto w = ct.vec(m);
	auto alpha = ct.val(1e-7);

	// load some data
	la::mat F_data(n, m);
	la::vec t_data(n, 0.0f);
	la::vec w_data(m, 0.0f);
	for (int i=0; i<n; i++) {
		for (int j=0; j<m; j++) {
			F_data[i][j] = rand() * 1.0f / RAND_MAX;
			t_data[i] += j * F_data[i][j];
		}
	}

	F.set(F_data);
	t.set(t_data);
	w.set(w_data);

	// train w

	auto FT_temp = F.T();
	auto FT = ct.mat(m, n);
	FT = std::move(FT_temp);

	stopwatch sw(0);

	for (int i=0; i<1 * 1024; i++) {
		auto tmp = F.dot(w) - t;
		w = w - FT.dot(tmp) * alpha;
	}

	sw.tock();

	// write output
	auto w_out = w.get();
	for (int i=0; i<10; i++)
		std::cout << w_out[i] << ' ';
	std::cout << '\n';
}


void simple_test() {
	auto ct = iopp::opencl_context();

	const int SZ = 1 << 22;

	la::vec a_data(SZ); 
	la::vec b_data(SZ);

	std::iota(a_data.begin(), a_data.end(), 1);
	std::iota(b_data.begin(), b_data.end(), 0);
	// std::reverse(b_data.begin(), b_data.end());

	auto a = ct.vec(SZ);
	auto b = ct.vec(SZ);
	auto c = ct.vec(SZ);

	a.set(a_data);
	b.set(b_data);

	stopwatch sw(0);

	for (int i=0; i<1024; i++) {
		c = a - b;
	}

	sw.tock();

	auto c_out = c.get();
	for (int i=0; i<5; i++)
		std::cout << c_out[SZ - 1 - i] << " ";
	std::cout << '\n';

	// std::cout << a.get() << '\n';
	// std::cout << b.get() << '\n';
	// std::cout << c.get() << '\n';
}


int main() {
	compile_check();
	// simple_test();
	medium_test();
}