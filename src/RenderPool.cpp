#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <glm/glm.hpp>

#include "Scene.h"
#include "Integrator.h"

#include "RenderPool.h"

RenderJob::RenderJob(glm::uvec2 startPixel, glm::uvec2 windowSize, int sampleIdx)
    : startPixel(startPixel),
      windowSize(windowSize),
	  sampleIdx(sampleIdx),
      idxSeq(NULL),
      startIdx(0),
      endIdx(0),
      _result((size_t)windowSize.x * windowSize.y)
{
}

RenderJob::RenderJob(std::vector<size_t>* _seqIdx, size_t _startIdx, size_t _endIdx)
    : startPixel(0),
      windowSize(0),
      sampleIdx(0),
      idxSeq(_seqIdx),
      startIdx(_startIdx),
      endIdx(_endIdx),
      _result(_endIdx - _startIdx)
{
}

void RenderJob::render(Scene* scene, Integrator* integrator)
{
    if (idxSeq != NULL)
    {
        for (size_t i = startIdx; i < endIdx; i++)
        {
            int x = idxSeq->at(i) % scene->imageSize.x;
            int y = idxSeq->at(i) / scene->imageSize.x;
            float biasX = getRandNum();
            float biasY = getRandNum();

            glm::vec3 target =
                scene->camera.imagePlaneTopLeft
                + (x + biasX) * scene->camera.pixelRight
                + (y + biasY) * scene->camera.pixelDown;
            glm::vec3 direction = glm::normalize(target - scene->camera.origin);

            _result[i - startIdx] = integrator->traceRay(scene->camera.origin, direction);
        }
        return;
    }

	for (size_t wy = 0; wy < windowSize.y; wy++) {
		size_t y = startPixel.y + wy;
		for (size_t wx = 0; wx < windowSize.x; wx++) {
			size_t x = startPixel.x + wx;

			glm::vec3 target;
			glm::vec3 direction;
			if (sampleIdx == 0)
			{
				target =
					scene->camera.imagePlaneTopLeft
					+ (x + 0.5f) * scene->camera.pixelRight
					+ (y + 0.5f) * scene->camera.pixelDown;
			}
			else
			{
				float biasX = getRandNum();
				float biasY = getRandNum();
				target =
					scene->camera.imagePlaneTopLeft
					+ (x + biasX) * scene->camera.pixelRight
					+ (y + biasY) * scene->camera.pixelDown;
			}

			direction = glm::normalize(target - scene->camera.origin);
			_result[wy * windowSize.x + wx] = integrator->traceRay(scene->camera.origin, direction);
		}
	}
}

std::vector<glm::vec3> RenderJob::getResult()
{
    return std::move(_result);
}

RenderPool::RenderPool(Scene* scene, Integrator* integrator, int numThreads, std::vector<RenderJob*>& jobs)
    : _scene(scene), _integrator(integrator), _nextJob(0), _jobQueue(jobs)
{
    for (int i = 0; i < numThreads; i++) {
        _threads.push_back(std::thread(threadMain, this));
    }
}

RenderPool::~RenderPool()
{
    for (std::thread& thread : _threads) {
        thread.join();
    }
}

void RenderPool::getCompletedJobs(std::vector<RenderJob*>& completedJobs)
{
    {
        std::unique_lock<std::mutex> lock(_mutex);

        _condition.wait(lock, [this]{ return _completedJobs.size() > 0; });
        completedJobs = std::move(_completedJobs);
    }
}

void RenderPool::threadMain(RenderPool* pool)
{
    while (true) {

        size_t jobIndex;
        {
            std::unique_lock<std::mutex> lock(pool->_mutex);

            if (pool->_nextJob >= pool->_jobQueue.size()) break;

            jobIndex = pool->_nextJob;
            pool->_nextJob++;
        }

        pool->_jobQueue[jobIndex]->render(pool->_scene, pool->_integrator);

        {
            std::unique_lock<std::mutex> lock(pool->_mutex);

            pool->_completedJobs.push_back(pool->_jobQueue[jobIndex]);
            pool->_condition.notify_all();
        }
    }
}
