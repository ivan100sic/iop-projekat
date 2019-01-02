#define LOCAL_SIZE 64
#define LOCAL_SIZE_SQRT 8
#define BLOCK_SIZE 1
#define LOOP int i = get_global_id(0) * BLOCK_SIZE, j; for (j=i; j<i+BLOCK_SIZE; j++) if (j < n)

// u + v

kernel void vadd(
	global float* a,
	global float* b,
	global float* c,
	int n
) {
	LOOP
		c[j] = a[j] + b[j];
}

kernel void vsub(
	global float* a,
	global float* b,
	global float* c,
	int n
) {
	LOOP
		c[j] = a[j] - b[j];
}

kernel void vmul(
	global float* a,
	global float* b,
	global float* c,
	int n
) {
	LOOP
		c[j] = a[j] * b[j];
}

kernel void vdiv(
	global float* a,
	global float* b,
	global float* c,
	int n
) {
	LOOP
		c[j] = a[j] / b[j];
}

// u += v

kernel void vaddc(
	global float* a,
	global float* b,
	int n
) {
	LOOP
		a[j] += b[j];
}

kernel void vsubc(
	global float* a,
	global float* b,
	int n
) {
	LOOP
		a[j] -= b[j];
}

kernel void vmulc(
	global float* a,
	global float* b,
	int n
) {
	LOOP
		a[j] *= b[j];
}

kernel void vdivc(
	global float* a,
	global float* b,
	int n
) {
	LOOP
		a[j] *= b[j];
}

// u + x

kernel void vsadd(
	global float* a,
	global float* b,
	float y,
	int n
) {
	LOOP
		b[j] = a[j] + y;
}

kernel void vssub(
	global float* a,
	global float* b,
	float y,
	int n
) {
	LOOP
		b[j] = a[j] - y;
}

kernel void vsmul(
	global float* a,
	global float* b,
	float y,
	int n
) {
	LOOP
		b[j] = a[j] * y;
}

kernel void vsdiv(
	global float* a,
	global float* b,
	float y,
	int n
) {
	LOOP
		b[j] = a[j] / y;
}

// u += x

kernel void vsaddc(
	global float* a,
	float y,
	int n
) {
	LOOP
		a[j] += y;
}

kernel void vssubc(
	global float* a,
	float y,
	int n
) {
	LOOP
		a[j] -= y;
}

kernel void vsmulc(
	global float* a,
	float y,
	int n
) {
	LOOP
		a[j] *= y;
}

kernel void vsdivc(
	global float* a,
	float y,
	int n
) {
	LOOP
		a[j] /= y;
}

// matrix ops

kernel void mt(
	global float* a,
	global float* b,
	int n,
	int m
) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (i < n && j < m) {
		b[j + i * m] = a[i + j * n];
	}
}

kernel void mvdot(
	global float* a,
	global float* b,
	global float* c,
	int n,
	int m
) {
	int i = get_global_id(0), j;
	float z = 0.0f;
	for (j = 0; j < m; j++) {
		z += a[i + j*n] * b[j];
	}
	c[i] = z;
}

kernel void mmdot(
	global float* a,
	global float* b,
	global float* c,
	int n,
	int m,
	int l
) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k;
	float z = 0.0f;
	for (k=0; k<m; k++) {
		z += a[i + k*n] * b[k + j*m];
	}
	c[i + j*n] = z;
}

kernel void vvouter(
	global float* a,
	global float* b,
	global float* c,
	int n,
	int m
) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	c[i + n*j] = a[i] * b[j];
}

kernel void rdsum_1(
	global float* a,
	global float* b,
	int n,
	int m
) {
	int i = get_global_id(0), j;
	float z = 0.0f;
	for (j=i; j<n; j+=m) {
		z += a[j];
	}
	b[i] = z;
}

kernel void rdsum_2(
	global float* a,
	int n
) {
	int i;
	float z = 0.0f;
	for (i=0; i<n; i++) {
		z += a[i];
	}
	a[0] = z;
}

// vector functions

kernel void vsqrtc(
	global float* a,
	int n
) {
	LOOP
		a[j] = sqrt(a[j]);
}

kernel void vexpc(
	global float* a,
	int n
) {
	LOOP
		a[j] = exp(a[j]);
}