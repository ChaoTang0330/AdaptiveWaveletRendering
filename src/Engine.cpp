#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <numeric>

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <embree3/rtcore.h>
#include <lodepng/lodepng.h>

#include "Scene.h"
#include "SceneLoader.h"
#include "Integrator.h"
#include "RenderPool.h"

#include "Engine.h"
#include "Wavelet.h"

#include <ctime>
#define _CRT_SECURE_NO_WARNINGS

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration = std::chrono::duration<float>;

const unsigned int WINDOW_DIM = 32;
const unsigned int SAMPLES_PER_JOB = 1024;

static unsigned char convertColorChannel(float channel)
{
    return static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, 255.0f * channel)));
}

static void saveImage(
    const std::vector<glm::vec3>& imageData,
    glm::uvec2 imageSize,
    const std::string& fileName)
{
	time_t now = time(0);
	tm* currTime = localtime(&now);
	std::string compFileName = "Result/" + fileName + "_" + std::to_string(currTime->tm_mon + 1)
		+ "_" + std::to_string(currTime->tm_mday)
		+ "_" + std::to_string(currTime->tm_hour)
		+ "_" + std::to_string(currTime->tm_min)
		+ ".png";

    std::vector<unsigned char> imageByteData(imageSize.y * imageSize.x * 3);
    for (size_t y = 0; y < imageSize.y; y++) {
        for (size_t x = 0; x < imageSize.x; x++) {

            glm::vec3 color = imageData[y * imageSize.x + x];

            size_t outPixelBase = 3 * (y * imageSize.x + x);
            imageByteData[outPixelBase + 0] = convertColorChannel(color.r);
            imageByteData[outPixelBase + 1] = convertColorChannel(color.g);
            imageByteData[outPixelBase + 2] = convertColorChannel(color.b);
        }
    }

    unsigned int error = lodepng::encode(
		compFileName, //fileName,
        imageByteData,
        imageSize.x,
        imageSize.y,
        LCT_RGB);
    if (error) {
        throw std::runtime_error(
            "LodePNG error (" + std::to_string(error) + "): "
            + lodepng_error_text(error));
    }
}

static void saveRawData(
    const std::vector<std::vector<glm::vec3>>& imageData,
    const std::string& fileName)
{
    time_t now = time(0);
    tm* currTime = localtime(&now);
    std::string compFileName = "Raw/" + fileName + "_" + std::to_string(currTime->tm_mon + 1)
        + "_" + std::to_string(currTime->tm_mday)
        + "_" + std::to_string(currTime->tm_hour)
        + "_" + std::to_string(currTime->tm_min)
        + ".csv";

    std::ofstream rawFile(compFileName);
    if (!rawFile.is_open()) throw std::runtime_error("Could not open file: '" + compFileName + "'");

    rawFile.precision(6);
    for (auto it = imageData.begin(); it != imageData.end(); it++)
    {
        rawFile << std::fixed << it->at(0).x << ", ";
        rawFile << std::fixed << it->at(0).y << ", ";
        rawFile << std::fixed << it->at(0).z;

        for (size_t i = 1; i < it->size(); i++)
        {
            rawFile << std::fixed << ", " << it->at(i).x << ", ";
            rawFile << std::fixed << it->at(i).y << ", ";
            rawFile << std::fixed << it->at(i).z;
        }

        rawFile << std::endl;
    }
}

static void embreeErrorFunction(void* userPtr, RTCError error, const char* str)
{
    (void) userPtr;
    std::cerr << "Embree error (" << error << "): " << str << std::endl;
}

static void printLoadingBar(float completion, int numBars = 60)
{
    int barsComplete = static_cast<int>(std::floor(numBars * completion));
    int percentComplete = static_cast<int>(std::floor(100 * completion));

    std::ostringstream oss;

    oss << "\r[";
    int j = 1;
    for (; j <= barsComplete; j++) {
        oss << '#';
    }
    for (; j <= numBars; j++) {
        oss << ' ';
    }
    oss << "] " << percentComplete << "%\r";

    std::cout << oss.str() << std::flush;
}

void render(const std::string& sceneFilePath)
{
    std::cout << "Loading scene..." << std::endl;

    //Initialize embree
    RTCDevice embreeDevice = rtcNewDevice(nullptr);
    if (!embreeDevice) throw std::runtime_error("Could not initialize Embree device.");
    rtcSetDeviceErrorFunction(embreeDevice, embreeErrorFunction, nullptr);

    //Load scene
    Scene* scene;
	Integrator* integrator;
    loadScene(sceneFilePath, embreeDevice, &scene, &integrator);

    //Loading wavelet coefficients
    std::cout << "Loading wavelet coefficients..." << std::endl;
    Wavelet* wl = new Wavelet("coefs_db97.txt", scene->imageSize.x);

	int numThreads = std::thread::hardware_concurrency();
    unsigned long sampleBudget = (unsigned long)scene->samplesPerPixel 
                                * scene->imageSize.x 
                                * scene->imageSize.y;
    unsigned long numCompletedSamples = 0;
    std::vector<std::vector<glm::vec3>> imageData((size_t)scene->imageSize.y * scene->imageSize.x);
    std::vector<glm::vec3> finalImage((size_t)scene->imageSize.y * scene->imageSize.x);
    std::cout << "Rendering... (" << sampleBudget << " samples, " 
        << numThreads << " threads)" << std::endl;
    
    TimePoint startTime = Clock::now();
    {
        // step 1: uniformally sampling, 4SPP
        //prepare jobs
        std::vector<RenderJob*> jobs;
        for (unsigned int y = 0; y < scene->imageSize.y; y += WINDOW_DIM)
        {
            for (unsigned int x = 0; x < scene->imageSize.x; x += WINDOW_DIM)
            {
                for (unsigned int i = 0; i < 4; i++)
                {
                    glm::uvec2 startPixel = glm::uvec2(x, y);
                    glm::uvec2 windowSize = glm::uvec2(
                        std::min(x + WINDOW_DIM, scene->imageSize.x) - x,
                        std::min(y + WINDOW_DIM, scene->imageSize.y) - y);
                    jobs.push_back(new RenderJob(startPixel, windowSize, i));
                }
            }
        }
        //execute job
        size_t numCompletedJobs = 0;
        RenderPool* pool = new RenderPool(scene, integrator, numThreads, jobs);
        while (numCompletedJobs < jobs.size()) {

            std::vector<RenderJob*> completedJobs;
            pool->getCompletedJobs(completedJobs);

            for (RenderJob* job : completedJobs) {
                std::vector<glm::vec3> result = job->getResult();
                for (unsigned long wy = 0; wy < job->windowSize.y; wy++) {
                    unsigned long y = job->startPixel.y + wy;
                    for (unsigned long wx = 0; wx < job->windowSize.x; wx++) {
                        unsigned long x = job->startPixel.x + wx;
						imageData[y * scene->imageSize.x + x].push_back(
							result[wy * job->windowSize.x + wx]);
                    }
                }
                numCompletedSamples += result.size();
            }
            numCompletedJobs += completedJobs.size();
            printLoadingBar(static_cast<double>(numCompletedSamples) / sampleBudget);
        }
        delete pool;
        
        // step 2: Adaptive sampling
        while (numCompletedSamples < sampleBudget)
        {
            jobs.clear();
            std::vector<size_t> idxSeq;
            wl->getPriorityPos(imageData, idxSeq);
            //prepare jobs
            for (size_t i = 0; i < idxSeq.size(); i += SAMPLES_PER_JOB)
            {
                size_t j = i + SAMPLES_PER_JOB;
                if (j <= idxSeq.size())
                {
                    jobs.push_back(new RenderJob(&idxSeq, i, j));
                }
                else
                {
                    jobs.push_back(new RenderJob(&idxSeq, i, idxSeq.size()));
                }
            }
            //excutive jobs
            numCompletedJobs = 0;
            pool = new RenderPool(scene, integrator, numThreads, jobs);
            while (numCompletedJobs < jobs.size()) 
            {
                std::vector<RenderJob*> completedJobs;
                pool->getCompletedJobs(completedJobs);

                for (RenderJob* job : completedJobs) 
                {
                    std::vector<glm::vec3> result = job->getResult();
                    for (size_t i = 0; i < result.size(); i++)
                    {
                        imageData[idxSeq[job->startIdx + i]].push_back(result[i]);
                    }
                    numCompletedSamples += result.size();
                }
                numCompletedJobs += completedJobs.size();
                printLoadingBar(static_cast<double>(numCompletedSamples) / sampleBudget);
            }
            delete pool;
        }
        wl->denoise(imageData, scene->gamma);
    }
    TimePoint endTime = Clock::now();
    Duration renderTime = endTime - startTime;

    //Save intermidiate result
    //saveRawData(imageData, scene->outputFileName);
    wl->showResult(scene->outputFileName);//

    std::cout << std::endl;
    std::cout << "Render time: " << renderTime.count() << "s" << std::endl;

    rtcReleaseScene(scene->embreeScene);
    rtcReleaseDevice(embreeDevice);

    //saveImage(finalImage, scene->imageSize, scene->outputFileName);
    std::cout << "Image saved as '" << scene->outputFileName << "'" << std::endl;
    delete wl;
}
