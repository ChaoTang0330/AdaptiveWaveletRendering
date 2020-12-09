#include "utils.hpp"
#include <tuple>

Mat scale_variance(Mat& sigmaB, Mat& phi,Mat& dst)
{
	Mat f = phi.mul(phi);
	f = 1.05 * (f / (sum(f)));
	Mat variance2,variance;
	filter2D(sigmaB, variance2,-1,f);
	variance = down_sample(variance2,false,false);
	dst = variance;
	return variance;
}

Mat down_sample(Mat& input, bool row_odd, bool col_odd)
{

	Mat image(input.rows / 2, input.cols / 2, CV_32F);
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			image.at<float>(i, j) = input.at<float>(2 * i + row_odd, 2 * j + col_odd);
		}
	}
	return image;

}

//template<typename T>
//Mat_<T> down_sampling(Mat_<T>& input, int colStart, int rowStart, int step)
//{
//	Mat_<T> image(input.rows / step, input.cols / step);
//	for (int y = colStart; y < input.rows; y += step)
//	{
//		for (int x = rowStart; x < input.cols; x += step)
//		{
//			image(y/step, x/step) = input(y, x);
//		}
//	}
//	return image;
//}
//
//template<typename T>
//Mat_<T> up_sampling(Mat_<T>& input, int colStart, int rowStart, int step)
//{
//	Mat_<T> image = cv::Mat_<T>::zeros(input.rows * step, input.cols * step);
//	for (int y = colStart; y < input.rows; y += step)
//	{
//		for (int x = rowStart; x < input.cols; x += step)
//		{
//			image(y, x) = input(y/step, x/step);
//		}
//	}
//	return image;
//}

Mat wavelet_magnitude(Mat& B, Mat& phi, Mat& psi0, Mat& psi1, Mat& psi2, Mat& ll)
{
	Mat ll0,w0, w1, w2,magnitude;
	filter2D(B, w0, -1, psi0);
	filter2D(B, w1, -1, psi1);
	filter2D(B, w2, -1, psi2);
	filter2D(B, ll0, -1, phi);
	Mat w01 = down_sample(w0, true, false);
	Mat w11 = down_sample(w1, false, true);
	Mat w21 = down_sample(w2, true, true);
	ll = down_sample(ll0, false, false);
	Mat temp1;
	add(1 / 3 * w01.mul(w01) , 1 / 3 * w11.mul(w11),temp1);
	add(temp1, 1 / 3 * w21.mul(w21),magnitude);
	return magnitude;
}


tuple<float, int, int> max_priority(Mat& image) {
	float max_priority = 0;
	int posx = 0;
	int posy = 0;
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{	
			float temp = *(float*)(image.data + image.step[0] * i + image.step[1] * j);
			if (temp > max_priority) {
				max_priority = temp;
				posx = i;float
				posy = j;
			}
		}
	}
	return make_tuple(max_priority, posx, posy);
}
