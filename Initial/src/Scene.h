#pragma once

#include <string>
#include <vector>
#include <limits>

#include <glm/glm.hpp>
#include <embree3/rtcore.h>

#include <map>

#include "Material.h"

struct camera_t {
    glm::vec3 origin;
    glm::vec3 imagePlaneTopLeft;
    glm::vec3 pixelRight;
    glm::vec3 pixelDown;
};

/*
struct material_t {
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
    glm::vec3 emission;
    glm::vec3 ambient;

	//HW4
	float roughness = 0.0f;
	int BRDFtype = 0;
};
*/

struct directionalLight_t {
    glm::vec3 toLight;
    glm::vec3 brightness;
};

struct pointLight_t {
    glm::vec3 point;
    glm::vec3 brightness;
    glm::vec3 attenuation;
};

//HW2
struct quadLight_t
{
	glm::vec3 a;
	glm::vec3 ab;
	glm::vec3 ac;
	glm::vec3 intensity;
};

class Scene {

public:

    glm::uvec2 imageSize;
    int maxDepth;
    std::string outputFileName;
    camera_t camera;
    std::vector<glm::mat3> sphereNormalTransforms;
    std::vector<material_t> sphereMaterials;
    std::vector<material_t> triMaterials;
    std::vector<directionalLight_t> directionalLights;
    std::vector<pointLight_t> pointLights;
    RTCScene embreeScene;

	//HW2
	std::vector<quadLight_t> quadLights;
	unsigned int lightsamples = 1;
	bool lightstratify = false;

	//HW3
	unsigned int samplesPerPixel;
	int NEEFlag = 0;
	bool RRFlag = false;

	//HW4
	int importanceSampling = 1;
	float gamma = 1.0f;
	std::map<int, int> lightGeoID;

	//Transmission
	std::vector <Material*> materials;
	std::vector <int> triMaterialID;
	std::vector <int> sphMaterialID;

	bool castRay(
		glm::vec3 origin,
		glm::vec3 direction,
		glm::vec3* hitPosition,
		glm::vec3* hitNormal,
		material_t* hitMaterial,
		int* hitGeoID = NULL) const; // HW4 modified

	bool castRay(
		glm::vec3 origin,
		glm::vec3 direction,
		glm::vec3* hitPosition,
		glm::vec3* hitNormal,
		Material** hitMaterial,
		int* hitGeoID = NULL) const; // HW4 modified

    bool castOcclusionRay(
        glm::vec3 origin,
        glm::vec3 direction,
        float maxDistance = std::numeric_limits<float>::infinity()) const;

};
