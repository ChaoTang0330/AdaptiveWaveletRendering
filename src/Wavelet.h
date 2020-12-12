#pragma once

#include<vector>
#include<opencv2/opencv.hpp>
#include <tuple>
#include <utility>
#include "glm\glm.hpp"

template<typename T>
void down_sampling(cv::Mat_<T>& input, int colStart, int rowStart, int step, cv::Mat_<T>& result);

template<typename T>
void up_sampling(cv::Mat_<T>& input, int colStart, int rowStart, int step, cv::Mat_<T>& result);

void imagesc(std::string title, cv::Mat_<float>& image);
void imageShow(std::string title, cv::Mat_<float>& image);

class Wavelet
{
private:
	int image_size = 1024;

	cv::Mat_<float> low;
	cv::Mat_<float> high;
	cv::Mat_<float> low_syn;
	cv::Mat_<float> high_syn;

	cv::Mat_<float> phi;
	cv::Mat_<float> psi[3];
	cv::Mat_<float> phi_syn;
	cv::Mat_<float> psi_syn[3];

	cv::Mat_<float> phi_square[5];
	cv::Mat_<float> psi_square[5][3];

	std::vector<float> scalingCDF[5];
	std::vector<float> low_k[5];
	std::vector<float> high_k[5];

	const float psi_square_norm = 0.707107;//
	const float phi_square_norm = 1.05;

	cv::Mat_<cv::Vec3f> finalImage;
	cv::Mat_<cv::Vec3f> pixMean;
	cv::Mat_<float> samplesDistri;

public:
	Wavelet() {}
	~Wavelet() {}
	Wavelet(std::string coefFile, int image_size);
	
	void readCoefs(std::string coefFile);
	void getMeanVar(std::vector<std::vector<glm::vec3>>& imageData,
		cv::Mat_<float>& mean_pix,
		cv::Mat_<float>& var_pix);
	void getRGBVar(std::vector<std::vector<glm::vec3>>& imageData,
		std::vector<cv::Mat_<float>>& var_pix);
	void getPriorityPos(std::vector<std::vector<glm::vec3>>& imageData,
		std::vector<size_t>& idxSeq);
	void sampleScaling(int level, 
		std::vector<std::pair<int, int>>& posSeq, 
		std::vector<size_t>& idxSeq);
	
	template<typename T>
	void DWT(cv::Mat_<T>& input, 
		cv::Mat_<T>& scalingCoefs, 
		std::vector<std::vector<cv::Mat_<T>>>& waveletCoefs);
	template<typename T>
	void IDWT(cv::Mat_<T>& scalingCoefs,
		std::vector<std::vector<cv::Mat_<T>>>& waveletCoefs,
		cv::Mat_<T>& result);

	void denoise(std::vector<std::vector<glm::vec3>>& imageData, float gamma);

	void getNLargestPri(cv::Mat_<float>& priority, size_t N,
		std::vector<std::pair<int, int>>& posSeq);

	void showResult(std::string fileName = "");
	void test(cv::Mat_<float>& image);
};