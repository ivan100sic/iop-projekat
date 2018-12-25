#define LOCAL_SIZE 16
#define LOCAL_SIZE_SQRT 4

kernel void vcopy(
	global float* a,
	global float* b,
	int n
) {
	int i = get_global_id(0) * 4, j;
	for (j=i; j<i+4; j++) {
		if (j < n) {
			b[j] = a[j];
		}
	}
}

kernel void vsub(
	global float* a,
	global float* b,
	global float* c,
	int n
) {
	int i = get_global_id(0) * 4, j;
	for (j=i; j<i+4; j++) {
		if (j < n) {
			c[j] = a[j] - b[j];
		}
	}
}

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

kernel void vsmul(
	global float* a,
	global float* b,
	float y,
	int n
) {
	int i = get_global_id(0) * 4, j;
	for (j=i; j<i+4; j++) {
		if (j < n) {
			b[j] = a[j] * y;
		}
	}
}

kernel void vsubc(
	global float* a,
	global float* b,
	int n
) {
	int i = get_global_id(0) * 4, j;
	for (j=i; j<i+4; j++) {
		if (j < n) {
			a[j] -= b[j];
		}
	}
}