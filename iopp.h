// #pragma once
// #define IOPP_ENABLE_OPENCL_LOG

#include "CL/cl.h"
#include "la.h"
#include <map>
#include <vector>
#include <string>

#define LOCAL_SIZE 64
#define LOCAL_SIZE_SQRT 8
#define BLOCK_SIZE 1

namespace iopp {

class _opencl_context;
class cl_vec;
class cl_val;
class cl_mat;

class cl_mat {
	friend class _opencl_context;
	friend class cl_vec;
	friend class cl_val;
protected:
	_opencl_context* context;
	cl_mem mem;
	int n, m;
	cl_mat(_opencl_context* context, cl_mem mem, int n, int m);
	void check(const cl_mat& b) const;
	void destroy();
public:
	cl_mat(const cl_mat& b);
	cl_mat(cl_mat&& b);
	cl_mat& operator= (const cl_mat& b);
	cl_mat& operator= (cl_mat&& b);
	~cl_mat();

	la::mat get() const;
	void set(const la::mat& a);

	cl_mat T() const;
	cl_vec dot(const cl_vec& v) const;
	cl_mat dot(const cl_mat& v) const;

	cl_mat operator+ (const cl_mat& b) const;
	cl_mat operator- (const cl_mat& b) const;
	cl_mat operator* (const cl_mat& b) const;
	cl_mat operator/ (const cl_mat& b) const;
	cl_mat& operator+= (const cl_mat& b);
	cl_mat& operator-= (const cl_mat& b);
	cl_mat& operator*= (const cl_mat& b);
	cl_mat& operator/= (const cl_mat& b);

	cl_mat operator+ (const cl_val& b) const;
	cl_mat operator- (const cl_val& b) const;
	cl_mat operator* (const cl_val& b) const;
	cl_mat operator/ (const cl_val& b) const;
	cl_mat& operator+= (const cl_val& b);
	cl_mat& operator-= (const cl_val& b);
	cl_mat& operator*= (const cl_val& b);
	cl_mat& operator/= (const cl_val& b);
};

class cl_vec {
	friend class _opencl_context;
	friend class cl_val;
	friend class cl_mat;
protected:
	_opencl_context* context;
	cl_mem mem;
	int n;
	cl_vec(_opencl_context* context, cl_mem mem, int n);
	void check(const cl_vec& b) const;
	void destroy();
public:
	cl_vec(const cl_vec& b);
	cl_vec(cl_vec&& b);
	cl_vec& operator= (const cl_vec& b);
	cl_vec& operator= (cl_vec&& b);
	~cl_vec();

	la::vec get() const;
	void set(const la::vec& v);
	void run_function(const char* fn);

	cl_val sum() const;
	cl_val dot(const cl_vec& b) const;
	cl_mat outer(const cl_vec& b) const;

	cl_vec operator+ (const cl_vec& b) const;
	cl_vec operator- (const cl_vec& b) const;
	cl_vec operator* (const cl_vec& b) const;
	cl_vec operator/ (const cl_vec& b) const;
	cl_vec& operator+= (const cl_vec& b);
	cl_vec& operator-= (const cl_vec& b);
	cl_vec& operator*= (const cl_vec& b);
	cl_vec& operator/= (const cl_vec& b);

	cl_vec operator+ (const cl_val& b) const;
	cl_vec operator- (const cl_val& b) const;
	cl_vec operator* (const cl_val& b) const;
	cl_vec operator/ (const cl_val& b) const;
	cl_vec& operator+= (const cl_val& b);
	cl_vec& operator-= (const cl_val& b);
	cl_vec& operator*= (const cl_val& b);
	cl_vec& operator/= (const cl_val& b);
};

class cl_val {
	friend class _opencl_context;
	friend class cl_vec;
	friend class cl_mat;
protected:
	_opencl_context* context;
	float val;
	cl_val(_opencl_context* context, float val);
public:
	float get() const;
};

class _opencl_context {
	friend class cl_mat;
	friend class cl_vec;
	friend class cl_val;
	friend _opencl_context opencl_context();
protected:
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	std::map<int, std::vector<cl_mem>> available_buffers;
	std::map<std::string, cl_kernel> kernel_cache;

	cl_platform_id get_platform();
	cl_device_id get_device(cl_platform_id platform);
	cl_context get_context(cl_platform_id platform);
	cl_command_queue get_command_queue(cl_context context, cl_device_id device);
	cl_program get_program(cl_device_id device, cl_context context);
	cl_kernel get_kernel(std::string name);
	_opencl_context();
	cl_mem new_buffer(int len);
	void recycle(int n, cl_mem mem);
	void mem_read(cl_mem src, void* dest, int n);
	void mem_write(const void* src, cl_mem dest, int n);
	void mem_copy(cl_mem src, cl_mem dest, int n);

	template<class T, class... U>
	void run_kernel_impl(std::string name, std::vector<int> dims,
		int cnt, T arg, U... args);

	void run_kernel_impl(std::string name, std::vector<int> dims, int cnt);
	
	template<class... T>
	void run_kernel(std::string name, std::vector<int> dims, T... args);

	// TODO: destructor, delete copy constructor, etc.
public:
	cl_mat mat(int n, int m);
	cl_vec vec(int n);
	cl_val val(float f);
};

_opencl_context opencl_context();

cl_vec sqrt(const cl_vec& v);
cl_vec exp(const cl_vec& v);
cl_vec relu(const cl_vec& v);
cl_vec relu_d(const cl_vec& v);
cl_vec tanh(const cl_vec& v);
cl_vec tanh_d(const cl_vec& v);

} // end namespace iopp