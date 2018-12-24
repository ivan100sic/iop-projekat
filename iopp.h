// #pragma once
#include "CL/cl.h"
#include "la.h"
#include <map>
#include <vector>
#include <string>

namespace iopp {

class _opencl_context;

class cl_mat {
	friend _opencl_context;
protected:
	_opencl_context* context;
	cl_mem mem;
	int n, m;
	cl_mat(_opencl_context* context, cl_mem mem, int n, int m);
};

class cl_vec {
	friend _opencl_context;
protected:
	_opencl_context* context;
	cl_mem mem;
	int n;
	cl_vec(_opencl_context* context, cl_mem mem, int n);
public:
	void set(const la::vec& v);
	la::vec get() const;
	cl_vec operator- (const cl_vec& b) const;
	~cl_vec();
};

class _opencl_context {
	friend class cl_mat;
	friend class cl_vec;
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
};

_opencl_context opencl_context();

} // end namespace iopp