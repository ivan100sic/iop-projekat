#pragma once
/*
	Vectors and matrices
	Vectors are columns (n x 1), its dimension is the number of rows
	Matrices are stored in row-major fashion, its dimensions are (rows, cols)
*/
#include <initializer_list>
#include <iostream>

namespace la {

template<class T>
class _mat;

template<class T>
class _vec {
protected:
	int n;
	T* a;

	void check_dims(const _vec& b) const {
		if (n != b.n)
			throw "operand size mismatch";
	}

public:

	// Generic OOP stvari

	_vec() : n(0), a(nullptr) {}

	_vec(int n) : n(n), a(new T[n]) {}

	_vec(int n, const T& val) : n(n), a(new T[n]) {
		for (int i=0; i<n; i++)
			a[i] = val;
	}

	~_vec() {
		delete[] a;
	}

	_vec(const _vec& b) : n(b.n), a(new T[n]) {
		for (int i=0; i<n; i++)
			a[i] = b.a[i];
	}

	_vec(_vec&& b) : n(b.n), a(b.a) {
		b.a = nullptr;
		b.n = 0;
	}

	template<class U>
	_vec(std::initializer_list<U> b) : n(b.size()), a(new T[n]) {
		auto it = b.begin();
		int i = 0;
		while (it != b.end()) {
			a[i++] = *it;
			++it;
		}
	}

	_vec& operator= (const _vec& b) {
		if (&b == this)
			return *this;

		delete[] a;
		n = b.n;
		a = new T[n];
		for (int i=0; i<n; i++)
			a[i] = b.a[i];
		return *this;
	}

	_vec& operator= (_vec&& b) {
		delete[] a;
		n = b.n;
		a = new T[n];
		for (int i=0; i<n; i++)
			a[i] = b.a[i];
		b.a = nullptr;
		b.a = 0;
		return *this;
	}

	// Generic C++ STL-like functions

	int size() const { return n; }

	bool empty() const { return n == 0; }

	const T* begin() const { return a; }

	const T* end() const { return a+n; }

	T* begin() { return a; }

	T* end() { return a+n; }

	T& operator[] (int i) { return a[i]; }
	const T& operator[] (int i) const { return a[i]; }

	// Scalar compound operators

	_vec& operator+= (const T& x) {
		for (int i=0; i<n; i++)
			a[i] += x;
		return *this;
	}

	_vec& operator-= (const T& x) {
		for (int i=0; i<n; i++)
			a[i] -= x;
		return *this;
	}

	_vec& operator*= (const T& x) {
		for (int i=0; i<n; i++)
			a[i] *= x;
		return *this;
	}

	_vec& operator/= (const T& x) {
		for (int i=0; i<n; i++)
			a[i] /= x;
		return *this;
	}

	// Scalar operators

	_vec operator+ (const T& x) const {
		_vec tmp = *this;
		tmp += x;
		return tmp;
	}

	_vec operator- (const T& x) const {
		_vec tmp = *this;
		tmp -= x;
		return tmp;
	}

	_vec operator* (const T& x) const {
		_vec tmp = *this;
		tmp *= x;
		return tmp;
	}

	_vec operator/ (const T& x) const {
		_vec tmp = *this;
		tmp /= x;
		return tmp;
	}

	// Vector component-wise compound operators
	_vec& operator+= (const _vec& x) {
		check_dims(x);
		for (int i=0; i<n; i++)
			a[i] += x.a[i];
		return *this;
	}

	_vec& operator-= (const _vec& x) {
		check_dims(x);
		for (int i=0; i<n; i++)
			a[i] -= x.a[i];
		return *this;
	}

	_vec& operator*= (const _vec& x) {
		check_dims(x);
		for (int i=0; i<n; i++)
			a[i] *= x.a[i];
		return *this;
	}

	_vec& operator/= (const _vec& x) {
		check_dims(x);
		for (int i=0; i<n; i++)
			a[i] /= x.a[i];
		return *this;
	}

	// Vector component-wise operators
	_vec operator+ (const _vec& x) const {
		_vec tmp = *this;
		tmp += x;
		return tmp;
	}

	_vec operator- (const _vec& x) const {
		_vec tmp = *this;
		tmp -= x;
		return tmp;
	}

	_vec operator* (const _vec& x) const {
		_vec tmp = *this;
		tmp *= x;
		return tmp;
	}

	_vec operator/ (const _vec& x) const {
		_vec tmp = *this;
		tmp /= x;
		return tmp;
	}

	_vec operator- () const {
		return *this * -1;
	}

	// Scalar product
	T inner(const _vec& x) const {
		check_dims(x);
		T z = 0;
		for (int i=0; i<n; i++)
			z += a[i] * x.a[i];
		return z;
	}

	// Alias for "inner"
	T dot(const _vec& x) const {
		return inner(x);
	}

	// Outer product of vectors, the result is a matrix
	_mat<T> outer(const _vec& x) const {
		if (n == 0 || x.n == 0)
			return _mat<T>();

		_mat<T> z(n, x.n);

		for (int i=0; i<n; i++)
			for (int j=0; j<x.n; j++)
				z[i][j] = a[i] * x.a[j];

		return z;
	}
};

template<class T>
std::ostream& operator<< (std::ostream& os, const _vec<T>& v) {
	os << "[";
	for (int i=0; i<v.size(); i++) {
		os << v[i];
		if (i+1 != v.size())
			os << ", ";
	}
	return os << "]";
}

template<class U>
class _mat {
protected:
	_vec<_vec<U>> a;

	void check_dims(const _mat& b) const {
		if (rows() != b.rows() || cols() != b.cols())
			throw "operand size mismatch";
	}

	class _vec_proxy {
	private:
		_vec<U>& v;
	public:
		friend class _mat;

		_vec_proxy(_vec<U>& v) : v(v) {}

		U& operator[] (int i) {
			return v[i];
		}
	};

	class const__vec_proxy {
	private:
		const _vec<U>& v;
	public:
		friend class _mat;

		const__vec_proxy(const _vec<U>& v) : v(v) {}

		const U& operator[] (int i) const {
			return v[i];
		}
	};

public:
	_mat() : a() {}

	_mat(int n, int m) : a(n, _vec<U>(m)) {}

	_mat(int n, int m, const U& val) :
		a(n, _vec<U>(m, val)) {}

	int rows() const { return a.size(); }

	int cols() const {
		if (rows() == 0)
			return 0;
		return a[0].size();
	}

	_mat(std::initializer_list<_vec<U>> b) {
		if (b.size() == 0)
			return;
		
		auto it0 = b.begin();
		auto it1 = it0;
		++it1;

		while (it1 != b.end()) {
			if (it0->size() != it1->size())
				throw "row size mismatch";
			++it0;
			++it1;
		}

		if (b.begin()->size() == 0)
			return;

		it0 = b.begin();
		int i = 0;

		a = _vec<_vec<U>>(b.size());
		while (it0 != b.end()) {
			a[i++] = *it0;
			++it0;
		}

	}

	int size() const { return rows() * cols(); }

	bool empty() const { return size() == 0; }

	_vec_proxy operator[] (int i) { return a[i]; }
	const__vec_proxy operator[] (int i) const { return a[i]; }

	// Scalar compound operators

	_mat& operator+= (const U& x) {
		for (int i=0; i<rows(); i++)
			for (int j=0; j<cols(); j++)
				a[i][j] += x;
		return *this;
	}

	_mat& operator-= (const U& x) {
		for (int i=0; i<rows(); i++)
			for (int j=0; j<cols(); j++)
				a[i][j] -= x;
		return *this;
	}

	_mat& operator*= (const U& x) {
		for (int i=0; i<rows(); i++)
			for (int j=0; j<cols(); j++)
				a[i][j] *= x;
		return *this;
	}

	_mat& operator/= (const U& x) {
		for (int i=0; i<rows(); i++)
			for (int j=0; j<cols(); j++)
				a[i][j] /= x;
		return *this;
	}

	// Scalar operators

	_mat operator+ (const U& x) const {
		_mat tmp = *this;
		tmp += x;
		return tmp;
	}

	_mat operator- (const U& x) const {
		_mat tmp = *this;
		tmp -= x;
		return tmp;
	}

	_mat operator* (const U& x) const {
		_mat tmp = *this;
		tmp *= x;
		return tmp;
	}

	_mat operator/ (const U& x) const {
		_mat tmp = *this;
		tmp /= x;
		return tmp;
	}

	// Matrix component-wise compound operators
	_mat& operator+= (const _mat& x) {
		check_dims(x);
		for (int i=0; i<rows(); i++)
			for (int j=0; j<cols(); j++)
				a[i][j] += x.a[i][j];
		return *this;
	}

	_mat& operator-= (const _mat& x) {
		check_dims(x);
		for (int i=0; i<rows(); i++)
			for (int j=0; j<cols(); j++)
				a[i][j] -= x.a[i][j];
		return *this;
	}

	_mat& operator*= (const _mat& x) {
		check_dims(x);
		for (int i=0; i<rows(); i++)
			for (int j=0; j<cols(); j++)
				a[i][j] *= x.a[i][j];
		return *this;
	}

	_mat& operator/= (const _mat& x) {
		check_dims(x);
		for (int i=0; i<rows(); i++)
			for (int j=0; j<cols(); j++)
				a[i][j] /= x.a[i][j];
		return *this;
	}

	// Matrix component-wise operators
	_mat operator+ (const _mat& x) const {
		_mat tmp = *this;
		tmp += x;
		return tmp;
	}

	_mat operator- (const _mat& x) const {
		_mat tmp = *this;
		tmp -= x;
		return tmp;
	}

	_mat operator* (const _mat& x) const {
		_mat tmp = *this;
		tmp *= x;
		return tmp;
	}

	_mat operator/ (const _mat& x) const {
		_mat tmp = *this;
		tmp /= x;
		return tmp;
	}

	_mat dot(const _mat& x) const {
		if (empty() || x.empty())
			return _mat();

		if (cols() != x.rows())
			throw "operand size mismatch";

		_mat tmp(rows(), x.cols(), 0);
		for (int i=0; i<rows(); i++)
			for (int j=0; j<cols(); j++)
				for (int k=0; k<x.cols(); k++)
					tmp[i][k] += a[i][j] * x[j][k];

		return tmp;
	}

	_vec<U> dot(const _vec<U>& x) const {
		if (empty() || x.empty())
			return _vec<U>();

		if (cols() != x.size())
			throw "operand size mismatch";

		_vec<U> tmp(rows(), (U)0);
		for (int i=0; i<rows(); i++)
			for (int j=0; j<cols(); j++)
				tmp[i] += a[i][j] * x[j];

		return tmp;
	}

	_mat T() const {
		if (empty())
			return _mat();

		_mat tmp(cols(), rows());
		for (int i=0; i<rows(); i++)
			for (int j=0; j<cols(); j++)
				tmp[j][i] = a[i][j];

		return tmp;
	}

	static _mat id(int n) {
		_mat t(n, n, 0);
		for (int i=0; i<n; i++)
			t[i][i] = 1;
		return t;
	}

	template<class V>
	friend std::ostream& operator<< (std::ostream& os, const _mat<V>& v);

};

template<class T>
std::ostream& operator<< (std::ostream& os, const _mat<T>& v) {
	if (v.empty())
		return os << "[]";

	os << '\n';
	os << "[" << v.a[0] << ",\n";
	for (int i=1; i<v.rows(); i++) {
		os << ' ' << v.a[i];
		if (i+1 != v.rows())
			os << ",\n";
		else
			os << "]\n";
	}
	return os;
}

typedef _vec<float> vec;
typedef _mat<float> mat;

} // end namespace la

