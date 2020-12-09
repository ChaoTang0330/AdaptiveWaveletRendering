#include<vector>
#include<opencv2/opencv.hpp>
#include<tuple>
using namespace std;
using namespace cv;

Mat down_sample(Mat& input, bool row_odd, bool col_odd);

Mat scale_variance(Mat& sigmaB,Mat& phi,Mat& dst);

Mat wavelet_magnitude(Mat& B, Mat& phi, Mat& psi0, Mat& psi1, Mat& psi2,Mat& ll);

tuple<float, int, int> max_priority(Mat& input);

//template<typename T>
//Mat_<T> down_sampling(Mat_<T>& input, int colStart, int rowStart, int step);
//
//template<typename T>
//Mat_<T> up_sampling(Mat_<T>& input, int colStart, int rowStart, int step);