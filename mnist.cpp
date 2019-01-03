#include "iopp.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;
using namespace iopp;
using namespace la;

namespace mnist {

auto ct = opencl_context();

vector<pair<int, vector<int>>> read_data(string fn) {
	ifstream ifs(fn);
	vector<pair<int, vector<int>>> result;
	while (ifs) {
		string ln;
		getline(ifs, ln);
		if (ln.size() == 0)
			return result;
		for (char& x : ln)
			if (x == ',')
				x = ' ';
		istringstream iss(ln);
		int label;
		iss >> label;
		vector<int> data;
		int x;
		while (iss >> x)
			data.push_back(x);
		result.push_back({label, data});
	}
	return result;
}

vector<pair<vec, vec>> preprocess(vector<pair<int, vector<int>>> data) {
	vector<pair<vec, vec>> result;
	for (auto p : data) {
		vec x(p.second.size());
		copy(p.second.begin(), p.second.end(), x.begin());
		vec t(10, 0.0f);
		t[p.first] = 1.0f;
		result.push_back({x / 256, t});		
	}
	return result;
}

struct mnist_model {
	cl_mat A, B;
	cl_vec c, d, x, t;
	cl_vec k, l, m, n, o, p;
	cl_val q;

	mat random_mat(int n, int m, float lo, float hi) {
		mat a(n, m);
		for (int i=0; i<n; i++) {
			for (int j=0; j<m; j++) {
				a[i][j] = lo + (hi - lo) * rand() / RAND_MAX;
			}
		}
		return a;
	}

	vec random_vec(int n, float lo, float hi) {
		vec a(n);
		for (int i=0; i<n; i++) {
			a[i] = lo + (hi - lo) * rand() / RAND_MAX;
		}
		return a;
	}

	void mwrite(ostream& os, const mat& a) {
		os.precision(10);
		os << fixed;
		for (int i=0; i<a.rows(); i++)
			for (int j=0; j<a.cols(); j++)
				os << a[i][j] << ' ';
	}

	void mread(istream& is, mat& a) {
		for (int i=0; i<a.rows(); i++)
			for (int j=0; j<a.cols(); j++)
				is >> a[i][j];
	}

	void vwrite(ostream& os, const vec& a) {
		os.precision(10);
		os << fixed;
		for (float f : a)
			os << f << ' ';
	}

	void vread(istream& is, vec& a) {
		for (float& f : a)
			is >> f;
	}

	void save(string fn) {
		ofstream ofs(fn);
		mwrite(ofs, A.get());
		mwrite(ofs, B.get());
		vwrite(ofs, c.get());
		vwrite(ofs, d.get());
	}

	void load(string fn) {
		ifstream ifs(fn);

		mat AA = A.get();
		mread(ifs, AA);
		A.set(AA);

		mat BB = B.get();
		mread(ifs, BB);
		B.set(BB);

		vec cc = c.get();
		vread(ifs, cc);
		c.set(cc);

		vec dd = d.get();
		vread(ifs, dd);
		d.set(dd);
	}

	mnist_model() :
		A(ct.mat(800, 784)),
		B(ct.mat(10, 800)),
		c(ct.vec(800)),
		d(ct.vec(10)),

		x(ct.vec(784)),
		t(ct.vec(10)),

		k(ct.vec(800)),
		l(ct.vec(800)),
		m(ct.vec(800)),
		n(ct.vec(10)),
		o(ct.vec(10)),
		p(ct.vec(10)),

		q(ct.val(0))
	{
		A.set(random_mat(800, 784, -0.01, 0.01));
		B.set(random_mat(10, 800, -0.01, 0.01));
		c.set(random_vec(800, -0.01, 0.01));
		d.set(random_vec(10, -0.01, 0.01));
	}

	void check_matrix(cl_mat& a) {
		bool bad = 0;
		auto aa = a.get();
		for (int i=0; i<(int)aa.rows(); i++) {
			for (int j=0; j<(int)aa.cols(); j++) {
				if (isnan(aa[i][j])) {
					bad = 1;
				}
			}
		}
		if (bad) {
			cerr << "NAN!\n";
		} else {
			cerr << "OK.\n";
		}
	}

	void print_span(cl_vec& a) {
		auto aa = a.get();
		float lo = aa[0], hi = aa[0];
		for (float f : aa)
			lo = min(lo, f), hi = max(hi, f);
		cerr << "span: " << lo << ' ' << hi << '\n';
	}

	float feed_forward(pair<vec, vec> data) {
		x.set(data.first);
		t.set(data.second);

		k = A.dot(x);
		l = k + c;
		m = tanh(l);
		n = B.dot(m);
		o = n + d;
		p = softmax(o);
		q = p.dot(t);

		return q.get();
	}

	cl_vec softmax(const cl_vec& o) {
		cl_vec tmp = exp(o);
		return tmp / tmp.sum();
	}

	cl_vec softmax_jacobian_dot(cl_vec& o, cl_vec& t) {
		auto s = softmax(o);
		return s * t - s * (s.dot(t));
	}

	// maksimizujemo q pa je zato + gradijent
	void back_propagate(float rate, float reg) {
		auto g1 = ct.vec(10);
		auto g2 = ct.vec(800);
		auto g3 = ct.vec(784);
		auto e = ct.val(rate);
		auto rg = ct.val(1.0f - reg);

		g1 = t;
		g1 = softmax_jacobian_dot(o, t);
		d += g1 * e;
		g2 = B.T().dot(g1);
		B += g1.outer(m) * e;
		g2 *= tanh_d(l);
		c += g2 * e;
		A += g2.outer(x) * e;

		A *= rg;
		B *= rg;
		c *= rg;
		d *= rg;
	}

	void print_diag() {
		// check_matrix(A);
		// check_matrix(B);
		// print_span(k);
		// print_span(m);

		// cerr << d.get() << '\n';
		cerr << p.get() << '\n';
		{
			auto tt = t.get();
			cerr << max_element(tt.begin(), tt.end()) - tt.begin() << ' ';
		}
		cerr << q.get() << '\n';
		cerr << '\n';
	}
};

}

void train() {
	srand(3211);
	cerr << setw(9) << fixed;
	using namespace mnist;
	auto r = preprocess(read_data("mnist_train.csv"));
	cerr << "testcases: " << r.size() << '\n';
	random_shuffle(r.begin(), r.end());

	mnist_model model;
	// model.load("model_main");

	// {
	// 	cl_vec t = ct.vec(3);
	// 	cl_vec w = ct.vec(3);
	// 	t.set({-1, 0, 2});
	// 	w.set({1, 10, 100});
	// 	cerr << model.softmax_jacobian_dot(t, w).get() << '\n';
	// }

	int acc_acc = 0;
	float gain_acc = 0;

	for (int i=0; i<1200000; i++) {
		model.feed_forward(r[i % r.size()]);
		model.back_propagate(1e-2, 1e-4);
		float t = model.q.get();
		gain_acc += t;
		if (t > 0.5f) {
			acc_acc++;
		}
		if (i % 501 == 0) {
			cerr << "epoch: " << i << '\n';
			cerr << "acc_acc: " << acc_acc << '\n';
			cerr << "gain_acc: " << gain_acc / 501 << '\n';
			acc_acc = 0;
			gain_acc = 0;
			model.print_diag();
		}
	}

	model.save("model_main");
}

void test() {
	using namespace mnist;

	mnist_model model;
	model.load("model_main");

	int acc_acc = 0;

	auto r = preprocess(read_data("mnist_test.csv"));

	for (int i=0; i<(int)r.size(); i++) {
		model.feed_forward(r[i]);
		auto d1 = model.p.get();
		auto d2 = r[i].second;
		int y = max_element(d1.begin(), d1.end()) - d1.begin();
		int t = max_element(d2.begin(), d2.end()) - d2.begin();

		if (y == t) {
			acc_acc++;
		}

		if (i % 501 == 0) {
			cerr << "accuracy: " << acc_acc << "/" << i+1 << '\n';
		}
	}

	cerr << "accuracy: " << acc_acc << "/" << r.size() << '\n';
}

int main() {
	test();
}