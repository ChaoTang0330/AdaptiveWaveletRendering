#pragma once

#include <glm/glm.hpp>

#include "Scene.h"
#include <algorithm>

#include <iostream>

/*
#include <thread>
#define PI 3.1415927f
#define RAND_MULTIPLIER 6364136223846793005ULL
#define RNAD_INCREMENT 1442695040888963407ULL

static float getRandNum()
{
	thread_local unsigned long long seed = std::hash<std::thread::id>{}(std::this_thread::get_id());
	seed = (RAND_MULTIPLIER * seed) + RNAD_INCREMENT;
	return static_cast<float>(seed & 0xffffffff) / 0xffffffff;
}
*/
class Integrator {

protected:

    Scene* _scene;

public:

    void setScene(Scene* scene)
    {
        _scene = scene;
    }

    virtual glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction) = 0;

};


class RayTracerIntegrator : public Integrator {

private:

    glm::vec3 computeShading(
        glm::vec3 incidentDirection,
        glm::vec3 toLight,
        glm::vec3 normal,
        glm::vec3 lightBrightness,
        const material_t& material);

    glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction, int depth);

public:

    virtual glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction);

};

//HW2
class AnalyticDirectIntegrator : public Integrator
{
private:
	glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction, int depth);
public:
	virtual glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction);
};

class DirectIntegrator : public Integrator
{
private:
	glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction, int depth);
	void getSampleRay(glm::vec3 origin, quadLight_t light, 
		std::vector<glm::vec3>* sampleDirect, std::vector<float>* sampleLength);
public:
	virtual glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction);
};


//HW3
class PathTracerIntegrator : public Integrator
{
private:
	glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction, int depth, glm::vec3 throughPut, 
		bool NEESwitch, int* lightID = NULL, double *hitDis = NULL);
	void sampleHemisphere(glm::vec3 normal,
		std::vector<glm::vec3>* sampleDirect);
	void sampleQuadLight(glm::vec3 origin, quadLight_t light,
		std::vector<glm::vec3>* sampleDirect, std::vector<float>* sampleLength);

	glm::vec3 directIntensity(glm::vec3 direction, glm::vec3 hitPosition, 
		glm::vec3 hitNormal, Material* hitMaterial);
	glm::vec3 nextIntensity(glm::vec3 direction, glm::vec3 hitPosition,glm::vec3 hitNormal, 
		Material* hitMaterial, glm::vec3 throughPut, bool NEESwitch, int depth);

	/*
	glm::vec3 reflactIntensity(glm::vec3 direction, glm::vec3 hitPosition,glm::vec3 hitNormal, 
		Material* hitMaterial, glm::vec3 throughPut, bool NEESwitch, int depth);

	glm::vec3 refractIntensity(glm::vec3 direction, glm::vec3 hitPosition, glm::vec3 refractDir, glm::vec3 hitNormal, 
		Material* hitMaterial, glm::vec3 throughPut, bool NEESwitch, int depth);
*/
	//HW4
	void sampleCosine(glm::vec3 normal,
		std::vector<glm::vec3>* sampleDirect);
	double pdfLight(int lightID, glm::vec3 omega_i, float distance);

public:
	virtual glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction);
};