// #pragma once
#include "iopp.h"
#include <cmath>

namespace iopp {

static void check_dims(int n, int m) {
	if (n != m)
		throw "operand size mismatch";
}

static int threads1d(int n) {
	return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}


//
// cl_mat
//





cl_mat::cl_mat(_opencl_context* context, cl_mem mem, int n, int m)
	: context(context), mem(mem), n(n), m(m) {}

void cl_mat::check(const cl_mat& b) const {
	if (!(n == b.n && m == b.m))
		throw "operand size mismatch";
	if (!(context == b.context))
		throw "operand context mismatch";
}

void cl_mat::destroy() {
	if (context && mem) {
		context->recycle(n*m*sizeof(float), mem);
		mem = NULL;
	}
}

cl_mat::cl_mat(const cl_mat& b) : context(b.context),
	mem(b.context->new_buffer(b.n*b.m*sizeof(float))), n(b.n), m(b.m)
{}

cl_mat::cl_mat(cl_mat&& b) : context(b.context), mem(b.mem), n(b.n), m(b.m) {
	b.destroy();
}

cl_mat& cl_mat::operator= (const cl_mat& b) {
	if (this != &b) {
		check(b);
		context->mem_copy(b.mem, mem, n*m*sizeof(float));
	}
	return *this;
}

cl_mat& cl_mat::operator= (cl_mat&& b) {
	if (this != &b) {
		check(b);
		mem = b.mem;
		b.destroy();
	}
	return *this;
}

cl_mat::~cl_mat() {
	destroy();
}

la::mat cl_mat::get() const {
	la::mat a(n, m);
	float* buff = new float[n * m];
	context->mem_read(mem, buff, n*m*sizeof(float));
	for (int i=0; i<n; i++) {
		for (int j=0; j<m; j++) {
			a[i][j] = buff[i + j*n];
		}
	}
	delete[] buff;
	return a;
}

void cl_mat::set(const la::mat& a) {
	check_dims(n, a.rows());
	check_dims(m, a.cols());
	float* buff = new float[n * m];
	for (int i=0; i<n; i++) {
		const auto& row = a[i];
		for (int j=0; j<m; j++) {
			buff[i + j*n] = row[j];
		}
	}
	context->mem_write(buff, mem, n*m*sizeof(float));
	delete[] buff;
}

cl_mat cl_mat::T() const {
	auto r = context->mat(m, n);
	context->run_kernel("mt", {n, m}, mem, r.mem, n, m);
	return r;
}

cl_vec cl_mat::dot(const cl_vec& v) const {
	check_dims(m, v.n);
	auto r = context->vec(n);
	context->run_kernel("mvdot", {n}, mem, v.mem, r.mem, n, m);
	return r;
}

cl_mat cl_mat::dot(const cl_mat& v) const {
	check_dims(m, v.n);
	auto r = context->mat(n, v.m);
	context->run_kernel("mmdot", {n, v.m}, mem, v.mem, r.mem, n, m, v.m);
	return r;
}



cl_mat cl_mat::operator+(const cl_mat& b) const {
	check(b);
	auto r = context->mat(b.n, b.m);
	context->run_kernel("vadd", {threads1d(n*m)}, mem, b.mem, r.mem, n*m);
	return r;
}

cl_mat cl_mat::operator-(const cl_mat& b) const {
	check(b);
	auto r = context->mat(b.n, b.m);
	context->run_kernel("vsub", {threads1d(n*m)}, mem, b.mem, r.mem, n*m);
	return r;
}

cl_mat cl_mat::operator*(const cl_mat& b) const {
	check(b);
	auto r = context->mat(b.n, b.m);
	context->run_kernel("vmul", {threads1d(n*m)}, mem, b.mem, r.mem, n*m);
	return r;
}

cl_mat cl_mat::operator/(const cl_mat& b) const {
	check(b);
	auto r = context->mat(b.n, b.m);
	context->run_kernel("vdiv", {threads1d(n*m)}, mem, b.mem, r.mem, n*m);
	return r;
}



cl_mat& cl_mat::operator+= (const cl_mat& v) {
	check(v);
	context->run_kernel("vaddc", {threads1d(n*m)}, mem, v.mem, n*m);
	return *this;
}

cl_mat& cl_mat::operator-= (const cl_mat& v) {
	check(v);
	context->run_kernel("vsubc", {threads1d(n*m)}, mem, v.mem, n*m);
	return *this;
}

cl_mat& cl_mat::operator*= (const cl_mat& v) {
	check(v);
	context->run_kernel("vmulc", {threads1d(n*m)}, mem, v.mem, n*m);
	return *this;
}

cl_mat& cl_mat::operator/= (const cl_mat& v) {
	check(v);
	context->run_kernel("vdivc", {threads1d(n*m)}, mem, v.mem, n*m);
	return *this;
}



cl_mat cl_mat::operator+ (const cl_val& v) const {
	auto r = context->mat(n, m);
	context->run_kernel("vsadd", {threads1d(n*m)}, mem, r.mem, v.val, n*m);
	return r;
}

cl_mat cl_mat::operator- (const cl_val& v) const {
	auto r = context->mat(n, m);
	context->run_kernel("vssub", {threads1d(n*m)}, mem, r.mem, v.val, n*m);
	return r;
}

cl_mat cl_mat::operator* (const cl_val& v) const {
	auto r = context->mat(n, m);
	context->run_kernel("vsmul", {threads1d(n*m)}, mem, r.mem, v.val, n*m);
	return r;
}

cl_mat cl_mat::operator/ (const cl_val& v) const {
	auto r = context->mat(n, m);
	context->run_kernel("vsdiv", {threads1d(n*m)}, mem, r.mem, v.val, n*m);
	return r;
}



cl_mat& cl_mat::operator+= (const cl_val& v) {
	context->run_kernel("vsaddc", {threads1d(n*m)}, mem, v.val, n*m);
	return *this;
}

cl_mat& cl_mat::operator-= (const cl_val& v) {
	context->run_kernel("vssubc", {threads1d(n*m)}, mem, v.val, n*m);
	return *this;
}

cl_mat& cl_mat::operator*= (const cl_val& v) {
	context->run_kernel("vsmulc", {threads1d(n*m)}, mem, v.val, n*m);
	return *this;
}

cl_mat& cl_mat::operator/= (const cl_val& v) {
	context->run_kernel("vsdivc", {threads1d(n*m)}, mem, v.val, n*m);
	return *this;
}



//
// cl_vec
//




void cl_vec::check(const cl_vec& b) const {
	if (!(n == b.n))
		throw "operand size mismatch";
	if (!(context == b.context))
		throw "operand context mismatch";
}

void cl_vec::destroy() {
	if (context && mem) {
		context->recycle(n*sizeof(float), mem);
		mem = NULL;
	}
}

cl_vec::cl_vec(const cl_vec& b) : context(b.context),
	mem(b.context->new_buffer(b.n*sizeof(float))), n(b.n) {}

cl_vec::cl_vec(cl_vec&& b) : context(b.context), mem(b.mem), n(b.n) {
	b.destroy();
}

cl_vec& cl_vec::operator= (const cl_vec& b) {
	if (this != &b) {
		check(b);
		context->mem_copy(b.mem, mem, n*sizeof(float));
	}
	return *this;
}

cl_vec& cl_vec::operator= (cl_vec&& b) {
	if (this != &b) {
		check(b);
		mem = b.mem;
		b.destroy();
	}
	return *this;
}

cl_vec::~cl_vec() {
	destroy();
}

la::vec cl_vec::get() const {
	la::vec v(n);
	context->mem_read(mem, v.begin(), n*sizeof(float));
	return v;
}

void cl_vec::set(const la::vec& v) {
	check_dims(n, v.size());
	context->mem_write(v.begin(), mem, n*sizeof(float));
}

void cl_vec::run_function(const char* fn) {
	context->run_kernel(fn, {threads1d(n)}, mem, n);
}

cl_val cl_vec::sum() const {
	int threads = std::max(LOCAL_SIZE, LOCAL_SIZE * (int)::sqrt(n / 512.0));
	cl_vec temp = context->vec(threads);
	context->run_kernel("rdsum_1", {threads}, mem, temp.mem, n, threads);
	context->run_kernel("rdsum_2", {}, temp.mem, threads);
	float x;
	context->mem_read(temp.mem, &x, sizeof(float));
	return context->val(x);
}

cl_val cl_vec::dot(const cl_vec& b) const {
	return (*this * b).sum();
}

cl_mat cl_vec::outer(const cl_vec& b) const {
	auto r = context->mat(n, b.n);
	context->run_kernel("vvouter", {n, b.n}, mem, b.mem, r.mem, n, b.n);
	return r;
}



cl_vec cl_vec::operator+(const cl_vec& b) const {
	check(b);
	auto r = context->vec(b.n);
	context->run_kernel("vadd", {threads1d(n)}, mem, b.mem, r.mem, n);
	return r;
}

cl_vec cl_vec::operator-(const cl_vec& b) const {
	check(b);
	auto r = context->vec(b.n);
	context->run_kernel("vsub", {threads1d(n)}, mem, b.mem, r.mem, n);
	return r;
}

cl_vec cl_vec::operator*(const cl_vec& b) const {
	check(b);
	auto r = context->vec(b.n);
	context->run_kernel("vmul", {threads1d(n)}, mem, b.mem, r.mem, n);
	return r;
}

cl_vec cl_vec::operator/(const cl_vec& b) const {
	check(b);
	auto r = context->vec(b.n);
	context->run_kernel("vdiv", {threads1d(n)}, mem, b.mem, r.mem, n);
	return r;
}



cl_vec& cl_vec::operator+= (const cl_vec& v) {
	check(v);
	context->run_kernel("vaddc", {threads1d(n)}, mem, v.mem, n);
	return *this;
}

cl_vec& cl_vec::operator-= (const cl_vec& v) {
	check(v);
	context->run_kernel("vsubc", {threads1d(n)}, mem, v.mem, n);
	return *this;
}

cl_vec& cl_vec::operator*= (const cl_vec& v) {
	check(v);
	context->run_kernel("vmulc", {threads1d(n)}, mem, v.mem, n);
	return *this;
}

cl_vec& cl_vec::operator/= (const cl_vec& v) {
	check(v);
	context->run_kernel("vdivc", {threads1d(n)}, mem, v.mem, n);
	return *this;
}



cl_vec cl_vec::operator+ (const cl_val& v) const {
	auto r = context->vec(n);
	context->run_kernel("vsadd", {threads1d(n)}, mem, r.mem, v.val, n);
	return r;
}

cl_vec cl_vec::operator- (const cl_val& v) const {
	auto r = context->vec(n);
	context->run_kernel("vssub", {threads1d(n)}, mem, r.mem, v.val, n);
	return r;
}

cl_vec cl_vec::operator* (const cl_val& v) const {
	auto r = context->vec(n);
	context->run_kernel("vsmul", {threads1d(n)}, mem, r.mem, v.val, n);
	return r;
}

cl_vec cl_vec::operator/ (const cl_val& v) const {
	auto r = context->vec(n);
	context->run_kernel("vsdiv", {threads1d(n)}, mem, r.mem, v.val, n);
	return r;
}



cl_vec& cl_vec::operator+= (const cl_val& v) {
	context->run_kernel("vsaddc", {threads1d(n)}, mem, v.val, n);
	return *this;
}

cl_vec& cl_vec::operator-= (const cl_val& v) {
	context->run_kernel("vssubc", {threads1d(n)}, mem, v.val, n);
	return *this;
}

cl_vec& cl_vec::operator*= (const cl_val& v) {
	context->run_kernel("vsmulc", {threads1d(n)}, mem, v.val, n);
	return *this;
}

cl_vec& cl_vec::operator/= (const cl_val& v) {
	context->run_kernel("vsdivc", {threads1d(n)}, mem, v.val, n);
	return *this;
}

//
// _opencl_context (i ostalo, trenutno)
//




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
		std::cerr << "allocating new buffer " << len << '\n';
		return clCreateBuffer(context, CL_MEM_READ_WRITE,
			len, NULL, NULL);
	} else {
		auto buff = available_buffers[len].back();
		available_buffers[len].pop_back();
		return buff;
	}
}

cl_vec::cl_vec(_opencl_context* context, cl_mem mem, int n)
	: context(context), mem(mem), n(n) {}

cl_mat _opencl_context::mat(int n, int m) {
	return cl_mat(this, new_buffer(n*m*sizeof(float)), n, m);
}

cl_vec _opencl_context::vec(int n) {
	return cl_vec(this, new_buffer(n*sizeof(float)), n);
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

_opencl_context opencl_context() {
	return _opencl_context();
}

void _opencl_context::recycle(int n, cl_mem mem) {
	available_buffers[n].push_back(mem);
}

cl_val::cl_val(_opencl_context* context, float val):
	context(context), val(val) {}

cl_val _opencl_context::val(float f) {
	return cl_val(this, f);
}

float cl_val::get() const {
	return val;
}

void _opencl_context::mem_copy(cl_mem src, cl_mem dest, int n) {
	clEnqueueCopyBuffer(queue, src, dest, 0, 0, n, 0, NULL, NULL);
	clFinish(queue);
}

void _opencl_context::mem_read(cl_mem src, void* dest, int n) {
	clEnqueueReadBuffer(queue, src, 1,
		0, n, dest,
		0, NULL, NULL);
	clFinish(queue);
}

void _opencl_context::mem_write(const void* src, cl_mem dest, int n) {
	clEnqueueWriteBuffer(queue, dest, 1,
		0, n, src,
		0, NULL, NULL);
	clFinish(queue);
}


//
// experiments etc
//



void sqrt(cl_vec& a) {
	a.run_function("vsqrtc");
}

void exp(cl_vec& a) {
	a.run_function("vexpc");
}


} // end namespace iopp