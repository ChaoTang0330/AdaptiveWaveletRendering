#include "priority_computation.hpp"
#include "glm\glm.hpp"
#include "utils.hpp"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>

#include "Constants.h"

#include <ctime>
#define _CRT_SECURE_NO_WARNINGS

#define OPENCV_TRAITS_ENABLE_DEPRECATED

const unsigned BASE_N = 8;
const unsigned int SAMPLES_LEVEL_FACTOR = 32;

template<typename T>
void down_sampling(cv::Mat_<T>& input, int colStart, int rowStart, int step, cv::Mat_<T>& result)
{
	result = cv::Mat_<T>::zeros(input.rows / step, input.cols / step);
	for (int y = colStart; y < input.rows; y += step)
	{
		for (int x = rowStart; x < input.cols; x += step)
		{
			result(y / step, x / step) = input(y, x);
		}
	}
}

template<typename T>
void up_sampling(cv::Mat_<T>& input, int colStart, int rowStart, int step, cv::Mat_<T>& result)
{
	result = cv::Mat_<T>::zeros(input.rows * step, input.cols * step);
	for (int y = colStart; y < result.rows; y += step)
	{
		for (int x = rowStart; x < result.cols; x += step)
		{
			result(y, x) = input(y / step, x / step);
		}
	}
}

void imagesc(std::string title, cv::Mat_<float>& image)
{
	double max, min;
	cv::minMaxIdx(image, &min, &max);
	cv::Mat adjMap;
	float scale = 255 / (max - 0);
	image.convertTo(adjMap, CV_8UC1, scale, 0);
	cv::Mat resultMap;
	applyColorMap(adjMap, resultMap, cv::COLORMAP_JET);
	cv::namedWindow(title, cv::WINDOW_NORMAL);
	cv::imshow(title, resultMap);
	cv::waitKey(0);
}

void imageShow(std::string title, cv::Mat_<float>& image)
{
	double max;
	cv::minMaxIdx(image, NULL, &max);
	cv::Mat adjMap;
	float scale = 255 / (max - 0);
	image.convertTo(adjMap, CV_8UC1, scale, 0);
	cv::namedWindow(title, cv::WINDOW_NORMAL);
	cv::imshow(title, adjMap);
	cv::waitKey(0);
}

Wavelet::Wavelet(std::string coefFile, int image_size)
{
	Wavelet::image_size = image_size;
	readCoefs(coefFile);

	phi = low * low.t();
	psi[0] = low * high.t();
	psi[1] = high * low.t();
	psi[2] = high * high.t();

	phi_syn = low_syn * low_syn.t();
	psi_syn[0] = low_syn * high_syn.t();
	psi_syn[1] = high_syn * low_syn.t();
	psi_syn[2] = high_syn * high_syn.t();

	//psi_square
	for (int i = 0; i < 5; i++)
	{
		cv::Mat_<float> curr_base(low_k[i].size(), 1, low_k[i].data());
		phi_square[i] = curr_base * curr_base.t();
		phi_square[i] = phi_square[i].mul(phi_square[i]);
		float mat_sum = cv::sum(phi_square[i])[0];
		phi_square[i] *= psi_square_norm / mat_sum;
	}

	//cdf
	for (int i = 0; i < 5; i++)
	{
		scalingCDF[i].resize(low_k[i].size());
		scalingCDF[i][0] = 0;
		for (size_t j = 1; j < low_k[i].size(); j++)
		{
			scalingCDF[i][j] = scalingCDF[i][j-1] + abs(low_k[i][j-1]);
		}
		float sum_mag = scalingCDF[i].back() + abs(low_k[i].back());
		for (size_t j = 0; j < low_k[i].size(); j++)
		{
			scalingCDF[i][j] /= sum_mag;
		}
	}
}

void Wavelet::readCoefs(std::string coefFile)
{
	std::ifstream file(coefFile);
	if (!file.is_open()) throw std::runtime_error("Could not open file: '" + coefFile + "'");

	std::string line;
	while (std::getline(file, line)) {
		std::istringstream tokenStream(line);

		std::string command;
		tokenStream >> command;

		if (command.size() == 0 || command[0] == '#') continue;

		if (command == "low")
		{
			std::vector<float> temp;
			while (!tokenStream.eof())
			{
				float currNum;
				tokenStream >> currNum;
				temp.push_back(currNum);
			}

			low = cv::Mat_<float>(temp.size(), 1, temp.data(), sizeof(float)).clone();
			low_k[0] = low.clone();
		}
		else if (command == "high")
		{
			std::vector<float> temp;
			while (!tokenStream.eof())
			{
				float currNum;
				tokenStream >> currNum;
				temp.push_back(currNum);
			}

			high = cv::Mat_<float>(temp.size(), 1, temp.data(), sizeof(float)).clone();
		}
		if (command == "low_syn")
		{
			std::vector<float> temp;
			while (!tokenStream.eof())
			{
				float currNum;
				tokenStream >> currNum;
				temp.push_back(currNum);
			}

			low_syn = cv::Mat_<float>(temp.size(), 1, temp.data(), sizeof(float)).clone();
		}
		else if (command == "high_syn")
		{
			std::vector<float> temp;
			while (!tokenStream.eof())
			{
				float currNum;
				tokenStream >> currNum;
				temp.push_back(currNum);
			}

			high_syn = cv::Mat_<float>(temp.size(), 1, temp.data(), sizeof(float)).clone();
		}
		else if (command == "low_k")
		{
			int currLevel;
			tokenStream >> currLevel;
			while (!tokenStream.eof())
			{
				float currNum;
				tokenStream >> currNum;
				low_k[currLevel - 1].push_back(currNum);
			}
		}
		else if (command == "high_ka")
		{
			int currLevel, currAlpha;
			tokenStream >> currLevel;
			while (!tokenStream.eof())
			{
				float currNum;
				tokenStream >> currNum;
				high_k[currLevel - 1].push_back(currNum);
			}
		}
	}
}

template<typename T>
void Wavelet::DWT(cv::Mat_<T>& input,
	cv::Mat_<T>& scalingCoefs,
	std::vector<std::vector<cv::Mat_<T>>>& waveletCoefs)
{
	scalingCoefs = input.clone();
	//imageShow("k = 0", scalingCoefs);
	for (size_t k = 0; k < waveletCoefs.size(); k++)
	{
		cv::Mat_<T> temp;
		for (int alpha = 0; alpha < 3; alpha++)
		{
			filter2D(scalingCoefs, temp, -1, psi[alpha]);
			down_sampling<T>(temp, (alpha + 1) >> 1, (alpha + 1) & 1, 2, waveletCoefs[k][alpha]);
		}
		filter2D(scalingCoefs, temp, -1, phi);
		down_sampling<T>(temp, 0, 0, 2, scalingCoefs);
		//imageShow("k = " + to_string(k + 1), scalingCoefs);
	}
}

template<typename T>
void Wavelet::IDWT(cv::Mat_<T>& scalingCoefs,
	std::vector<std::vector<cv::Mat_<T>>>& waveletCoefs,
	cv::Mat_<T>& result)
{
	result = scalingCoefs.clone();
	//imageShow("k = 5", result);
	for (int k = waveletCoefs.size() - 1; k >= 0; k--)//waveletCoefs.size() - 1
	{
		cv::Mat_<T> temp_up;
		up_sampling<T>(result, 0, 0, 2, temp_up);
		filter2D(temp_up, result, -1, phi_syn);

		for (int alpha = 0; alpha < 3; alpha++)
		{
			up_sampling<T>(waveletCoefs[k][alpha], (alpha + 1) >> 1, (alpha + 1) & 1, 2, temp_up);
			cv::Mat_<T> temp;
			filter2D(temp_up, temp, -1, psi_syn[alpha]);
			result += temp;
		}
		//imageShow("k = " + to_string(k), result);
	}
}

void Wavelet::getMeanVar(std::vector<std::vector<glm::vec3>>& imageData,
	cv::Mat_<float>& var_pix,
	cv::Mat_<float>& mean_pix)
{
	var_pix = cv::Mat_<float>::zeros(image_size, image_size);
	mean_pix = cv::Mat_<float>::zeros(image_size, image_size);

	for (int i = 0; i < imageData.size(); i++) {
		int x = i % image_size;
		int y = i / image_size;
		float max_val = -1;
		float min_val = 1;
		float sum = 0;
		for (int j = 0; j < imageData[i].size(); j++) {
			float value = (imageData[i][j].r + imageData[i][j].g + imageData[i][j].b) / 3;
			sum += value;
			max_val = max(max_val, value);
			min_val = min(min_val, value);
		}
		mean_pix(y, x) = min(sum / imageData[i].size(), (float)1.0);
		max_val = min(max_val, 1.0f);
		min_val = min(min_val, 1.0f);
		var_pix(y, x) =
			((max_val - min_val) * (max_val - min_val)) / ((max_val + min_val) * (max_val + min_val) + EPSILON)
			/ imageData[i].size();
	}
}

void  Wavelet::getPriorityPos(std::vector<std::vector<glm::vec3>>& imageData,
	std::vector<size_t>& idxSeq)
{
	cv::Mat_<float> var_pix, mean_pix;
	getMeanVar(imageData, var_pix, mean_pix);

	//test(mean_pix);
	//imageShow("var", var_pix);
	std::vector<std::pair<int, int>> posSeq;

	/*int level = 0;
	double max_pri = 0;
	cv::Point2i max_pos(0, 0);
	cv::minMaxLoc(var_pix, NULL, &max_pri, NULL, &max_pos);*/
	getNLargestPri(var_pix, 1 << BASE_N, posSeq);
	sampleScaling(0, posSeq, idxSeq);

	cv::Mat_<float> scalingCoefs;
	std::vector<std::vector<cv::Mat_<float>>> waveletCoefs(5, std::vector<cv::Mat_<float>>(3));
	DWT<float>(mean_pix, scalingCoefs, waveletCoefs);
	
	for (int k = 0; k < 5; k++)
	{
		cv::Mat_<float> var_filtered, var_scaling;
		filter2D(var_pix, var_filtered, -1, phi_square[k]);
		down_sampling<float>(var_filtered, 0, 0, 1 << (k + 1), var_scaling);
		cv::Mat_<float> priority = var_scaling -(
				waveletCoefs[k][0].mul(waveletCoefs[k][0])
				+ waveletCoefs[k][1].mul(waveletCoefs[k][1])
				+ waveletCoefs[k][2].mul(waveletCoefs[k][2])
			) / 3.0f;
		getNLargestPri(priority, 1 << (BASE_N - k), posSeq);
		sampleScaling(k + 1, posSeq, idxSeq);
		/*double curr_max_pri = 0;
		cv::Point2i curr_max_pos(0, 0);
		cv::minMaxLoc(priority, NULL, &curr_max_pri, NULL, &curr_max_pos);

		if (curr_max_pri >= max_pri)
		{
			max_pri = curr_max_pri;
			max_pos = curr_max_pos;
		}*/

		/******************/
		//imagesc("k = " + std::to_string(k), priority);
		//cout << endl;
		//cout << "k = " << k << endl;
		//cout << var_scaling.row(0) << endl;
		//cout << endl;
		/******************/
	}

	return;
}

void Wavelet::sampleScaling(int level, 
	std::vector<std::pair<int, int>>& posSeq, 
	std::vector<size_t>& idxSeq)
{
	if (level == 0)
	{
		for (auto [x, y] : posSeq)
		{
			size_t currIdx = x + y * image_size;
			for (int i = 0; i < SAMPLES_LEVEL_FACTOR; i++)
			{
				idxSeq.push_back(currIdx);
				/*if(currIdx > (1 << 20))
				{
					cout << "error" << endl;
				}*/
			}
		}
		return;
	}
	
	for (auto [x, y] : posSeq)
	{
		x = x << level;
		y = y << level;

		for (size_t i = 0; i < (SAMPLES_LEVEL_FACTOR << level); i++)
		{
			size_t x_sample, y_sample;
			do
			{
				float p = getRandNum();
				int idx = std::lower_bound(
					scalingCDF[level - 1].begin(),
					scalingCDF[level - 1].end(), p)
					- scalingCDF[level - 1].begin();
				idx -= scalingCDF[level - 1].size() / 2;
				x_sample = x + idx;
			} while (x_sample < 0 || x_sample >= image_size);

			do
			{
				float p = getRandNum();
				int idx = std::lower_bound(
					scalingCDF[level - 1].begin(),
					scalingCDF[level - 1].end(), p)
					- scalingCDF[level - 1].begin();
				idx -= scalingCDF[level - 1].size() / 2;
				y_sample = y + idx;
			} while (y_sample < 0 || y_sample >= image_size);

			idxSeq.push_back(x_sample + y_sample * image_size);
			/*if (x_sample + y_sample * image_size > (1 << 20))
			{
				cout << "error" << endl;
			}*/
		}
	}
}

void Wavelet::getNLargestPri(cv::Mat_<float>& priority, size_t N,
	std::vector<std::pair<int, int>>& posSeq)
{
	posSeq.clear();
	cv::Mat_<float> temp(1, priority.cols);
	for (int x = 0; x < priority.cols; x++) temp(0, x) = x;
	cv::Mat_<float> idx;
	cv::repeat(temp, priority.rows, 1, idx);
	cv::Mat_<cv::Vec3f> priWithIdx;
	cv::merge(vector<cv::Mat>{ priority, idx, idx.t() }, priWithIdx);
	priWithIdx = priWithIdx.reshape(0, 1);

	cv::Vec3f* start_add = priWithIdx.ptr<cv::Vec3f>(0, 0);
	cv::Vec3f* end_add = priWithIdx.ptr<cv::Vec3f>(0, priWithIdx.cols - 1) + 1;
	std::partial_sort(start_add, start_add + N, end_add, 
		[](cv::Vec3f& x, cv::Vec3f& y) -> bool {return x[0] > y[0]; });

	for (int i = 0; i < N; i++) 
		posSeq.push_back({ priWithIdx(0,i)[1],  priWithIdx(0,i)[2] });
}

void Wavelet::denoise(std::vector<std::vector<glm::vec3>>& imageData, float gamma)
//,std::vector<glm::vec3>& result)
{
	finalImage = cv::Mat_<cv::Vec3f>::zeros(image_size, image_size);
	samplesDistri = cv::Mat_<float>::zeros(image_size, image_size);

	//TO DO
	for (size_t i = 0; i < imageData.size(); i++)
	{
		glm::vec3 avgColor = std::accumulate(imageData[i].begin(), imageData[i].end(), glm::vec3(0));
		avgColor /= (float)imageData[i].size();

		int x = i % image_size;
		int y = i / image_size;
		//gamma correction
		finalImage(y, x)[2] = min(pow((double)avgColor.r, 1 / (double)gamma), 1.0);
		finalImage(y, x)[1] = min(pow((double)avgColor.g, 1 / (double)gamma), 1.0);
		finalImage(y, x)[0] = min(pow((double)avgColor.b, 1 / (double)gamma), 1.0);

		samplesDistri(y, x) = imageData[i].size();
	}
}

void Wavelet::showResult(std::string fileName)
{
	//show image
	cv::Mat reScaleImg;
	finalImage.convertTo(reScaleImg, CV_8UC3, 255, 0);
	cv::namedWindow("Final Image", cv::WINDOW_NORMAL);
	cv::imshow("Final Image", reScaleImg);
	cv::waitKey(0);

	double max;
	cv::minMaxIdx(samplesDistri, NULL, &max);
	cv::Mat adjMap;
	float scale = 255 / (max - 4);
	samplesDistri.convertTo(adjMap, CV_8UC1, scale, -255 / 4.0);
	cv::namedWindow("Sample Distributions", cv::WINDOW_NORMAL);
	cv::imshow("Sample Distributions", adjMap);
	cv::waitKey(0);

	if (fileName != "")
	{
		time_t now = time(0);
		tm* currTime = localtime(&now);
		std::string compFileName = "Result/" + fileName + "_" + std::to_string(currTime->tm_mon + 1)
			+ "_" + std::to_string(currTime->tm_mday)
			+ "_" + std::to_string(currTime->tm_hour)
			+ "_" + std::to_string(currTime->tm_min)
			+ ".png";
		std::string distFileName = "Result/" + fileName + "_Distribution_" + std::to_string(currTime->tm_mon + 1)
			+ "_" + std::to_string(currTime->tm_mday)
			+ "_" + std::to_string(currTime->tm_hour)
			+ "_" + std::to_string(currTime->tm_min)
			+ ".png";

		cv::imwrite(compFileName, reScaleImg);
		cv::imwrite(distFileName, adjMap);
	}
}

void Wavelet::test(cv::Mat_<float>& image)
{
	imageShow("Origin", image);
	cv::Mat_<float> scalingCoefs;
	std::vector<std::vector<cv::Mat_<float>>> waveletCoefs(5, std::vector<cv::Mat_<float>>(3));
	DWT<float>(image, scalingCoefs, waveletCoefs);

	cv::Mat_<float> result;
	IDWT<float>(scalingCoefs, waveletCoefs, result);
	imageShow("Recons", result);
}