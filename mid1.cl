// TODO: Add OpenCL kernel code here.

__kernel
void img_sobel(__global float *dest_data,
	__global float *src_data,
	int W, int H)
{
	int dest_x = get_global_id(0);
	int dest_y = get_global_id(1);
	
	if (dest_x >= W || dest_y >= H)	return;

	float m_y[9] = { -1.0f, -2.0f, -1.0f,
					 0.0f, 0.0f, 0.0f,
					 1.0f, 2.0f, 1.0f };

	float m_x[9] = { -1.0f, 0.0f, 1.0f,
					 -2.0f, 0.0f, 2.0f,
					 -1.0f, 0.0f, 1.0f };

	if(dest_x >= 1 && dest_x < W-1 && dest_y >=1 && dest_y < H-1)
	{ 
		float output_dy = 0.0f;		float output_dx = 0.0f;

		output_dy = (src_data[(dest_y - 1) * W + (dest_x - 1)] * m_y[0]
					+ src_data[(dest_y - 1) * W + dest_x] * m_y[1]
					+ src_data[(dest_y - 1) * W + (dest_x + 1)] * m_y[2]

					+ src_data[(dest_y) * W + (dest_x - 1)] * m_y[3]
					+ src_data[(dest_y) * W + dest_x] * m_y[4]
					+ src_data[(dest_y) * W + (dest_x + 1)] * m_y[5]

					+ src_data[(dest_y + 1)* W + (dest_x - 1)] * m_y[6]
					+ src_data[(dest_y + 1)* W + dest_x] * m_y[7]
					+ src_data[(dest_y + 1)* W + (dest_x + 1)] * m_y[8]);

		output_dx = (src_data[(dest_y - 1) * W + (dest_x - 1)] * m_y[0]
					+ src_data[(dest_y - 1) * W + dest_x] * m_y[1]
					+ src_data[(dest_y - 1) * W + (dest_x + 1)] * m_y[2]

					+ src_data[(dest_y)* W + (dest_x - 1)] * m_y[3]
					+ src_data[(dest_y)* W + dest_x] * m_y[4]
					+ src_data[(dest_y)* W + (dest_x + 1)] * m_y[5]

					+ src_data[(dest_y + 1)* W + (dest_x - 1)] * m_y[6]
					+ src_data[(dest_y + 1)* W + dest_x] * m_y[7]
					+ src_data[(dest_y + 1)* W + (dest_x + 1)] * m_y[8]);
	    
		float magnitude = sqrt( pow(output_dy, 2) + pow(output_dx, 2));
		
		if (magnitude > 255)
			magnitude = 255;
		else if (magnitude < 0)
			magnitude = 0;

		dest_data[dest_y*W + dest_x] = magnitude;
	}






}