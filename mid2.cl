double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

__kernel
void mat_mul(__global double* A, __global double *B, __global double *C, int COL_A)
{
	int y = get_global_id(0);	
	int x = get_global_id(1);	

	int COL_B = get_global_size(0);
	int ROW_A = get_global_size(1);

	double result = 0.0f;

	if (y < (COL_B) && x < (ROW_A)) 
	{
		for (int z = 0; z < COL_A; z++) 
		{
			result += A[x * COL_A + z] * B[z * COL_B + y];
		}
		C[x*COL_B + y] = sigmoid(result);
	}

}