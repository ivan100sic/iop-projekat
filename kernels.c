#define LOCAL_SIZE 16
#define LOCAL_SIZE_SQRT 4
#define LOOP int i = get_global_id(0) * 4, j; for (j=i; j<i+4; j++) if (j < n)

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

// matrix ops and others

kernel void mt(
	global float* a,
	global float* b,
	int n,
	int m
) {
	int i = get_global_id(0) * 2;
	int j = get_global_id(1) * 2;
	int ii, jj;
	for (ii=i; ii<i+2; ii++) {
		for (jj=j; jj<j+2; jj++) {
			if (ii < n && jj < m) {
				b[jj + ii*m] = a[ii + jj*n];
			}
		}
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