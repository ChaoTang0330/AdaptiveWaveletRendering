/*HW2*/

#include "Integrator.h"

/*************************AnalyticDirectIntegrator**********************************/
glm::vec3 AnalyticDirectIntegrator::traceRay(glm::vec3 origin, glm::vec3 direction)
{
	return traceRay(origin, direction, 0);
}

glm::vec3 AnalyticDirectIntegrator::traceRay(glm::vec3 origin, glm::vec3 direction, int depth)
{
	glm::vec3 outputColor = glm::vec3(0.0f, 0.0f, 0.0f);

	glm::vec3 hitPosition;
	glm::vec3 hitNormal;
	material_t hitMaterial;
	bool hit = _scene->castRay(origin, direction, &hitPosition, &hitNormal, &hitMaterial);
	if (hit) {
		//outputColor += hitMaterial.ambient;
		outputColor += hitMaterial.emission;
		
		for (const quadLight_t light : _scene->quadLights) 
		{
			glm::vec3 lightV[4];
			
			glm::vec3 pointA = light.a;
			glm::vec3 pointB = light.a + light.ab;
			glm::vec3 pointC = light.a + light.ac;
			glm::vec3 pointD = pointC + light.ab;

			lightV[0] = pointA - hitPosition;
			lightV[1] = pointB - hitPosition;
			lightV[3] = pointC - hitPosition;
			lightV[2] = pointD - hitPosition;

			glm::vec3 phi = glm::vec3(0, 0, 0);
			for (int i = 0; i < 4; i++)
			{
				glm::vec3 temp1 = lightV[i];
				glm::vec3 temp2 = lightV[(i + 1) % 4];
				float theta_i = glm::acos(glm::dot(glm::normalize(temp1), glm::normalize(temp2)));
				glm::vec3 gamma_i = glm::cross(temp1, temp2);
				gamma_i = glm::normalize(gamma_i);
				phi += theta_i * gamma_i;
			}
			phi = phi * 0.5f;
			float cosTheta = glm::dot(phi, hitNormal);
			glm::vec3 tempColor = hitMaterial.diffuse / PI * light.intensity * cosTheta;
			outputColor += tempColor;
		}
	}

	return outputColor;
}

/****************************DirectIntegrator**************************************/
glm::vec3 DirectIntegrator::traceRay(glm::vec3 origin, glm::vec3 direction)
{
	return traceRay(origin, direction, 0);
}

glm::vec3 DirectIntegrator::traceRay(glm::vec3 origin, glm::vec3 direction, int depth)
{
	glm::vec3 outputColor = glm::vec3(0.0f, 0.0f, 0.0f);

	glm::vec3 hitPosition;
	glm::vec3 hitNormal;
	material_t hitMaterial;
	bool hit = _scene->castRay(origin, direction, &hitPosition, &hitNormal, &hitMaterial);
	if (hit) {
		//outputColor += hitMaterial.ambient;
		outputColor += hitMaterial.emission;

		glm::vec3 reflectCamRay = glm::reflect(direction, hitNormal);
		for (const quadLight_t light : _scene->quadLights) // each light
		{
			glm::vec3 lightNorm = glm::cross(light.ab, light.ac);
			float quadLightArea = glm::length(lightNorm);
			lightNorm /= quadLightArea;

			std::vector<glm::vec3> sampleDirect;
			std::vector<float> sampleLength;
			getSampleRay(hitPosition, light, &sampleDirect, &sampleLength);

			glm::vec3 sumSample(0.0f);
			for (size_t i = 0; i < sampleDirect.size(); i++) // each sample
			{
				bool occluded = 
					_scene->castOcclusionRay(hitPosition, sampleDirect[i], sampleLength[i]);
				if (!occluded)
				{
					float specularFactor = glm::dot(reflectCamRay, sampleDirect[i]);
					specularFactor = std::max(specularFactor, 0.0f);

					glm::vec3 f_Phong = hitMaterial.specular *
						(0.5f * hitMaterial.shininess + 1.0f) * 
						pow(specularFactor, hitMaterial.shininess);
					f_Phong += hitMaterial.diffuse;
					f_Phong /= PI;

					float cosTheta_i = glm::dot(hitNormal, sampleDirect[i]);
					float cosTheta_o = glm::dot(lightNorm, sampleDirect[i]);
					cosTheta_i = std::max(cosTheta_i, 0.0f);
					cosTheta_o = std::max(cosTheta_o, 0.0f);

					sumSample += f_Phong * cosTheta_i * cosTheta_o 
						/ (sampleLength[i] * sampleLength[i]);
				}
			}

			glm::vec3 L_d = quadLightArea / (float)_scene->lightsamples * light.intensity * sumSample;
			outputColor += L_d;
		}
	}

	return outputColor;
}

void DirectIntegrator::getSampleRay(glm::vec3 origin, quadLight_t light,
	std::vector<glm::vec3>* sampleDirect, std::vector<float>* sampleLength)
{
	if (_scene->lightstratify)
	{
		int gridNum = sqrt((float)_scene->lightsamples);
		glm::vec3 gridUnitAB = light.ab / (float)gridNum;
		glm::vec3 gridUnitAC = light.ac / (float)gridNum;

		for (int i = 0; i < gridNum; i++)
		{
			for (int j = 0; j < gridNum; j++)
			{
				glm::vec3 samplePoint = light.a;
				samplePoint += ((float)i + getRandNum()) * gridUnitAB;
				samplePoint += ((float)j + getRandNum()) * gridUnitAC;
				glm::vec3 lightRayDir = samplePoint - origin;
				float lightRayLen = glm::length(lightRayDir);
				sampleDirect->push_back(lightRayDir / lightRayLen);
				sampleLength->push_back(lightRayLen);
			}
		}
	}
	else
	{
		for (unsigned int i = 0; i < _scene->lightsamples; i++)
		{
			glm::vec3 samplePoint = light.a;
			samplePoint += getRandNum() * light.ab;
			samplePoint += getRandNum() * light.ac;
			glm::vec3 lightRayDir = samplePoint - origin;
			float lightRayLen = glm::length(lightRayDir);
			sampleDirect->push_back(lightRayDir / lightRayLen);
			sampleLength->push_back(lightRayLen);
		}
	}
}