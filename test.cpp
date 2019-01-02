#include "iopp.h"
#include "la.h"
#include "stopwatch.h"
#include <numeric>
#include <algorithm>
#include <cmath>
// using namespace iopp;

auto ct = iopp::opencl_context();

void compile_check() {
	{
		auto v = ct.vec(10);
		v.set({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
		v = v - v * v / v + v;
		v.get();
		v += v -= v *= v /= v;

		auto a = ct.val(3.33);
		v = (v + a - a) * a / a;
		v += v -= v *= v /= a;
	}

	{
		auto v = ct.mat(3, 3);
		v.set({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
		v = v - v * v / v + v;
		std::cerr << (v * v + v).get();
		v += v -= v *= v /= v;
		v = v.dot(v);

		auto a = ct.val(3.33);
		v = (v + a - a) * a / a;
		v += v -= v *= v /= a;
	}

	{
		const int n = 13, m = 15;
		int k = 0;
		auto v = ct.mat(n, m);
		la::mat w(n, m);
		for (int i=0; i<n; i++)
			for (int j=0; j<m; j++)
				w[i][j] = k++;
		v.set(w);
		std::cerr << (v.T()).get();
	}

	{
		const int n = 1321;
		auto v = ct.vec(n);
		auto w = la::vec(n);
		std::iota(w.begin(), w.end(), 0);
		v.set(w);
		auto f = (v.dot(v)).get();
		std::cerr << "suma: " << f << '\n';
	}

	{
		auto a = ct.mat(3, 4);
		auto b = ct.mat(4, 2);
		auto c = ct.mat(3, 2);
		a.set({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
		b.set({{1, 2}, {3, 4}, {5, 6}, {7, 8}});
		std::cerr << a.dot(b).get() << '\n';
	}

	{
		auto a = ct.vec(4);
		auto b = ct.vec(3);
		a.set({1, 2, 3, 4});
		b.set({91, 108, -44});
		std::cerr << a.outer(b).get() << '\n';
	}
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

	auto FT = F.T();

	stopwatch sw(0);

	// auto bzvz = FT.dot(F);

	sw.tock();

	// train w
	for (int i=0; i<1 * 1024; i++) {
		auto tmp = F.dot(w) - t;
		w -= FT.dot(tmp) * alpha;
	}

	sw.tock();

	// write output
	auto w_out = w.get();
	for (int i=0; i<10; i++)
		std::cout << w_out[i] << ' ';
	std::cout << '\n';
}


void simple_test() {
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

	for (int i=0; i<4096; i++) {
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

void sqrt_test() {
	const int SZ = 1 << 22;

	la::vec a_data(SZ); 

	std::iota(a_data.begin(), a_data.end(), 1);
	auto a = ct.vec(SZ);
	auto c = ct.vec(SZ);
	a.set(a_data);

	stopwatch sw(0);

	for (int i=0; i<4096; i++) {
		c = a;
		iopp::exp(c);
	}

	sw.tock();

	auto c_out = c.get();
	for (int i=0; i<5; i++)
		std::cout << c_out[i] << " ";
	std::cout << '\n';
	for (int i=0; i<5; i++)
		std::cout << c_out[SZ - 1 - i] << " ";
	std::cout << '\n';

	// std::cout << a.get() << '\n';
	// std::cout << b.get() << '\n';
	// std::cout << c.get() << '\n';
}

void cpu_test() {
	const int SZ = 1 << 22;
	float* a = new float[SZ];
	float* c = new float[SZ];
	std::iota(a, a+SZ, 1);
	stopwatch sw(0);
	for (int i=0; i<32; i++) {
		std::copy(a, a+SZ, c);
		for (int j=0; j<SZ; j++)
			c[j] = exp(c[j]);
	}
	sw.tock();

	for (int i=0; i<5; i++)
		std::cout << c[i] << " ";
	std::cout << '\n';
	for (int i=0; i<5; i++)
		std::cout << c[SZ - 1 - i] << " ";

	delete[] a;
	delete[] c;
}

void transpose_test() {
	const int n = 8192, m = 4096;
	auto a = ct.mat(n, m);
	stopwatch sw(0);
	for (int i=0; i<64; i++) {
		auto b = a.T();
	}
	sw.tock();
}

void reduce_sum_test() {
	const int n = 1 << 22;
	auto v = ct.vec(n);
	auto w = la::vec(n);
	std::iota(w.begin(), w.end(), 0);
	v.set(w);
	auto u = v * v;
	auto f = ct.val(0.0);

	stopwatch sw(0);

	for (int i=0; i<1024; i++) {
		f = u.sum();
	}

	sw.tock();

	std::cerr << "suma: " << f.get() << '\n';
}

void outer_sum_test() {
	const int n = 8192;
	const int m = 4096;
	auto u = ct.vec(n);
	auto v = ct.vec(m);
	auto uu = la::vec(n);
	auto vv = la::vec(m);
	std::iota(uu.begin(), uu.end(), 0);
	std::iota(vv.begin(), vv.end(), 0);
	u.set(uu);
	v.set(vv);

	auto w = ct.mat(n, m);

	stopwatch sw(0);

	for (int i=0; i<1024; i++) {
		w = u.outer(v);
	}

	sw.tock();
}

int main() {
	compile_check();
	// simple_test();
	// medium_test();
	// sqrt_test();
	// cpu_test();
	// transpose_test();
	// reduce_sum_test();
	// outer_sum_test();
}