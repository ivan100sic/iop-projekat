// #pragma once
#include "iopp.h"

namespace iopp {

const int LOCAL_SIZE = 16;
const int LOCAL_SIZE_SQRT = 4;

static void check_dims(int n, int m) {
	if (n != m)
		throw "operand size mismatch";
}

cl_platform_id _opencl_context::get_platform() {
	cl_platform_id a[1];
	clGetPlatformIDs(1, a, NULL);
	return a[0];
}

cl_device_id _opencl_context::get_device(cl_platform_id platform) {
	cl_device_id a[1];
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, a, NULL);
	return a[0];
}

cl_context _opencl_context::get_context(cl_platform_id platform) {
	cl_context_properties properties[] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platform,
		0
	};
	return clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
}

cl_command_queue _opencl_context::get_command_queue(
	cl_context context, cl_device_id device
) {
	return clCreateCommandQueueWithProperties(context, device, 0, NULL);
}

cl_program _opencl_context::get_program(cl_device_id device, cl_context context) {
	const char* PATH = "kernels.c";
	FILE* f = fopen(PATH, "rb");
	fseek(f, 0, SEEK_END);
	long fsize = ftell(f);
	fseek(f, 0, SEEK_SET);
	char* source = new char[fsize + 1];
	fread(source, fsize, 1, f);
	fclose(f);
	source[fsize] = 0;

	cl_program program = clCreateProgramWithSource(
		context, 1, (const char**)&source, NULL, NULL
	);

	cl_device_id devices[] = {device};
	clBuildProgram(program, 1, devices, NULL, NULL, NULL);

	delete[] source;
	return program;
}

cl_kernel _opencl_context::get_kernel(std::string name) {
	if (!kernel_cache.count(name)) {
		int err;
		kernel_cache[name] = clCreateKernel(program, name.c_str(), &err);
		std::cerr << "kernel error " << name << ' ' << err << '\n';
	}
	return kernel_cache[name];
}

_opencl_context::_opencl_context() {
	platform = get_platform();
	device = get_device(platform);
	context = get_context(platform);
	queue = get_command_queue(context, device);
	program = get_program(device, context);
}

cl_mem _opencl_context::new_buffer(int len) {
	if (available_buffers[len].empty()) {
		return clCreateBuffer(context, CL_MEM_READ_WRITE,
			len * sizeof(cl_float), NULL, NULL);
	} else {
		auto buff = available_buffers[len].back();
		available_buffers[len].pop_back();
		return buff;
	}
}

cl_mat::cl_mat(_opencl_context* context, cl_mem mem, int n, int m)
	: context(context), mem(mem), n(n), m(m) {}

cl_vec::cl_vec(_opencl_context* context, cl_mem mem, int n)
	: context(context), mem(mem), n(n) {}

cl_mat _opencl_context::mat(int n, int m) {
	return cl_mat(this, new_buffer(n*m), n, m);
}

cl_vec _opencl_context::vec(int n) {
	return cl_vec(this, new_buffer(n), n);
}

void _opencl_context::run_kernel_impl(std::string name, std::vector<int> dims, int cnt) {
	size_t gws[2];
	size_t lws[2];
	int dc;

	if (dims.size() == 0) {
		gws[0] = lws[0] = dc = 1;
	} else if (dims.size() == 1) {
		gws[0] = (dims[0] + LOCAL_SIZE - 1) / LOCAL_SIZE * LOCAL_SIZE;
		lws[0] = LOCAL_SIZE;
		dc = 1;
	} else if (dims.size() == 2) {
		gws[0] = (dims[0] + LOCAL_SIZE_SQRT - 1) / LOCAL_SIZE_SQRT * LOCAL_SIZE_SQRT;
		gws[1] = (dims[1] + LOCAL_SIZE_SQRT - 1) / LOCAL_SIZE_SQRT * LOCAL_SIZE_SQRT;
		lws[0] = LOCAL_SIZE_SQRT;
		lws[1] = LOCAL_SIZE_SQRT;
		dc = 2;
	} else {
		throw "invalid number of dimensions";
	}

	// std::cerr << "lansiram sa parametrima: " << dc << ' ' << gws[0] << ' ' << lws[0] << '\n';

	clEnqueueNDRangeKernel(queue, get_kernel(name),
		dc, NULL, gws, lws,
		0, NULL, NULL);

	clFinish(queue);
}

template<class T, class... U>
void _opencl_context::run_kernel_impl(std::string name, std::vector<int> dims,
	int cnt, T arg, U... args
) {
	clSetKernelArg(get_kernel(name), cnt, sizeof(T), &arg);
	run_kernel_impl(name, dims, cnt+1, args...);
}

template<class... T>
void _opencl_context::run_kernel(std::string name, std::vector<int> dims, T... args) {
	run_kernel_impl(name, dims, 0, args...);
}

void cl_vec::set(const la::vec& v) {
	check_dims(n, v.size());
	clEnqueueWriteBuffer(context->queue, mem, 1,
		0, sizeof(float)*n, v.begin(),
		0, NULL, NULL);
	clFinish(context->queue);
}

cl_vec cl_vec::operator-(const cl_vec& b) const {
	check_dims(n, b.n);
	auto r = context->vec(b.n);
	context->run_kernel("vsub", {(n+3)/4}, mem, b.mem, r.mem, n);
	return r;
}

la::vec cl_vec::get() const {
	la::vec r(n);
	clEnqueueReadBuffer(context->queue, mem, 1,
		0, sizeof(float)*n, r.begin(),
		0, NULL, NULL);
	return r;
}

_opencl_context opencl_context() {
	return _opencl_context();
}

cl_vec::~cl_vec() {
	if (context) {
		context->recycle(n, mem);
	}
}

void _opencl_context::recycle(int n, cl_mem mem) {
	available_buffers[n].push_back(mem);
}

cl_vec& cl_vec::operator= (const cl_vec& b) {
	if (this != &b) {
		check_dims(n, b.n);
		context->run_kernel("vcopy", {(n+3)/4}, b.mem, mem, n);
	}
	return *this;
}

cl_val::cl_val(_opencl_context* context, float val):
	context(context), val(val) {}

cl_val _opencl_context::val(float f) {
	return cl_val(this, f);
}

cl_mat cl_mat::T() const {
	auto r = context->mat(m, n);
	context->run_kernel("mt", {(n+1)/2, (m+1)/2},
		mem, r.mem, n, m);
	return r;
}

cl_vec cl_mat::dot(const cl_vec& v) const {
	check_dims(m, v.n);
	auto r = context->vec(n);
	context->run_kernel("mvdot", {n}, mem, v.mem, r.mem, n, m);
	return r;
}

cl_vec cl_vec::operator* (const cl_val& v) const {
	auto r = context->vec(n);
	context->run_kernel("vsmul", {(n+3)/4}, mem, r.mem, v.val, n);
	return r;
}

cl_vec& cl_vec::operator-= (const cl_vec& v) {
	check_dims(n, v.n);
	context->run_kernel("vsubc", {(n+3)/4}, mem, v.mem, n);
	return *this;
}

void cl_mat::set(const la::mat& a) {
	check_dims(n, a.rows());
	check_dims(m, a.cols());
	float* buff = new float[n * m];
	for (int i=0; i<n; i++) {
		for (int j=0; j<m; j++) {
			buff[i + j*n] = a[i][j];
		}
	}
	clEnqueueWriteBuffer(context->queue, mem, 1,
		0, sizeof(float) * n * m, buff,
		0, NULL, NULL);
	clFinish(context->queue);
	delete[] buff;
}

la::mat cl_mat::get() const {
	la::mat a(n, m);
	float* buff = new float[n * m];
	clEnqueueReadBuffer(context->queue, mem, 1,
		0, sizeof(float) * n * m, buff,
		0, NULL, NULL);
	for (int i=0; i<n; i++) {
		for (int j=0; j<m; j++) {
			a[i][j] = buff[i + j*n];
		}
	}
	delete[] buff;
	return a;
}

cl_mat& cl_mat::operator= (const cl_mat& b) {
	if (this != &b) {
		check_dims(n, b.n);
		check_dims(m, b.m);
		context->run_kernel("vcopy", {(n*m+3)/4}, b.mem, mem, n*m);
	}
	return *this;
}

} // end namespace iopp