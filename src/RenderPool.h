#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <glm/glm.hpp>

#include "Scene.h"
#include "Integrator.h"

class RenderJob {

public:

    const glm::uvec2 startPixel;
    const glm::uvec2 windowSize;
	const int sampleIdx;

    const std::vector<size_t>* idxSeq;
    const size_t startIdx;
    const size_t endIdx;
private:

    std::vector<glm::vec3> _result;

public:

    RenderJob(glm::uvec2 startPixel, glm::uvec2 windowSize, int sampleIdx);
    RenderJob(std::vector<size_t>* _idxSeq, size_t _startIdx, size_t _endIdx);
    void render(Scene* scene, Integrator* integrator);
    std::vector<glm::vec3> getResult();

};

class RenderPool {

private:

    Scene* _scene;
    Integrator* _integrator;
    std::vector<std::thread> _threads;
    std::mutex _mutex;
    std::condition_variable _condition;
    size_t _nextJob;
    std::vector<RenderJob*>& _jobQueue;
    std::vector<RenderJob*> _completedJobs;


public:

    RenderPool(Scene* scene, Integrator* integrator, int numThreads, std::vector<RenderJob*>& jobs);
    ~RenderPool();
    void getCompletedJobs(std::vector<RenderJob*>& completedJobs);

private:

    static void threadMain(RenderPool* pool);

};
