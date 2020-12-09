#include <algorithm>

#include <glm/glm.hpp>

#include "Integrator.h"

glm::vec3 RayTracerIntegrator::computeShading(
    glm::vec3 incidentDirection,
    glm::vec3 toLight,
    glm::vec3 normal,
    glm::vec3 lightBrightness,
    const material_t& material)
{
    glm::vec3 diffuseReflectance = material.diffuse * std::max(glm::dot(normal, toLight), 0.0f);
    glm::vec3 halfAngle = glm::normalize(toLight + incidentDirection);
    glm::vec3 specularReflectance =
        material.specular
        * std::pow(std::max(glm::dot(normal, halfAngle), 0.0f), material.shininess);
    return lightBrightness * (diffuseReflectance + specularReflectance);
}

glm::vec3 RayTracerIntegrator::traceRay(glm::vec3 origin, glm::vec3 direction, int depth)
{
    glm::vec3 outputColor = glm::vec3(0.0f, 0.0f, 0.0f);

    glm::vec3 hitPosition;
    glm::vec3 hitNormal;
    material_t hitMaterial;
    bool hit = _scene->castRay(origin, direction, &hitPosition, &hitNormal, &hitMaterial);
    if (hit) {

        outputColor += hitMaterial.ambient;
        outputColor += hitMaterial.emission;

        for (const directionalLight_t light : _scene->directionalLights) {

            bool occluded = _scene->castOcclusionRay(hitPosition, light.toLight);
            if (!occluded) {
                outputColor += computeShading(
                    -direction,
                    light.toLight,
                    hitNormal,
                    light.brightness,
                    hitMaterial);
            }
        }

        for (const pointLight_t light : _scene->pointLights) {

            glm::vec3 toLight = light.point - hitPosition;
            float lightDistance = glm::length(toLight);
            toLight /= lightDistance;

            bool occluded = _scene->castOcclusionRay(hitPosition, toLight, lightDistance);
            if (!occluded) {
                float falloff =
                    light.attenuation.x
                    + light.attenuation.y * lightDistance
                    + light.attenuation.z * lightDistance * lightDistance;
                outputColor += computeShading(
                    -direction,
                    toLight,
                    hitNormal,
                    light.brightness / falloff,
                    hitMaterial);
            }
        }

        if (depth < _scene->maxDepth) {
            outputColor +=
                hitMaterial.specular
                * traceRay(hitPosition, glm::reflect(direction, hitNormal), depth + 1);
        }
    }

    return outputColor;
}

glm::vec3 RayTracerIntegrator::traceRay(glm::vec3 origin, glm::vec3 direction)
{
    return traceRay(origin, direction, 0);
}
