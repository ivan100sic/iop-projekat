// #pragma once
#include "CL/cl.h"
#include "la.h"
#include <map>
#include <vector>
#include <string>

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
public:
	cl_mat T() const;
	cl_vec dot(const cl_vec& v) const;
	void set(const la::mat& a);
	la::mat get() const;
	cl_mat& operator= (const cl_mat& b);
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
public:
	void set(const la::vec& v);
	la::vec get() const;
	cl_vec operator- (const cl_vec& b) const;
	cl_vec& operator= (const cl_vec& b);
	cl_vec& operator-= (const cl_vec& b);
	cl_vec operator* (const cl_val& y) const;
	~cl_vec();
};

class cl_val {
	friend class _opencl_context;
	friend class cl_vec;
	friend class cl_mat;
protected:
	_opencl_context* context;
	float val;
	cl_val(_opencl_context* context, float val);
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

} // end namespace iopp