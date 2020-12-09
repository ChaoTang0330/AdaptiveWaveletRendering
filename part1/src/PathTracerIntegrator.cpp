#include "Integrator.h"

glm::vec3 PathTracerIntegrator::traceRay(glm::vec3 origin, glm::vec3 direction)
{
	glm::vec3 throughPut(1, 1, 1);
	int depth = _scene->maxDepth;
	if (_scene->NEEFlag == 0) //NEE off
	{
		depth++;
		return traceRay(origin, direction, depth, throughPut, false);
	}
	else if (_scene->NEEFlag == 1) // NEE on
	{
		return traceRay(origin, direction, depth, throughPut, true);
	}
	else if (_scene->NEEFlag == 2) // MIS, NEE + BRDF 
	{
		glm::vec3 NEEColor = traceRay(origin, direction, depth, throughPut, true);//depth

		depth++;
		glm::vec3 BRDFColor = traceRay(origin, direction, depth, throughPut, false);

		return NEEColor + BRDFColor;
	}

	return glm::vec3(0);
}

glm::vec3 PathTracerIntegrator::traceRay(glm::vec3 origin, glm::vec3 direction, int depth,
	glm::vec3 throughPut, bool NEESwitch, int* lightID, double* hitDis)
{
	glm::vec3 outputColor = glm::vec3(0.0f, 0.0f, 0.0f);

	if (depth == 0 && !_scene->RRFlag) // reach max depth
	{
		return outputColor;
	}

	glm::vec3 hitPosition;
	glm::vec3 hitNormal;
	Material* hitMaterial;
	int hitGeoID = -1;
	bool hit = _scene->castRay(origin, direction, &hitPosition, &hitNormal, &hitMaterial, &hitGeoID);

	if (hit) {
		std::map<int, int>::iterator lightGeoIt;
		lightGeoIt = _scene->lightGeoID.find(hitGeoID);
		if (lightGeoIt != _scene->lightGeoID.end()) // hit light
		{
			if (glm::dot(direction, hitNormal) >= 0.0f)
			{
				outputColor = _scene->quadLights[lightGeoIt->second].intensity;
			}

			if (lightID != NULL)
			{
				*lightID = lightGeoIt->second;
				if (hitDis != NULL)
				{
					*hitDis = glm::length(hitPosition - origin);
				}
			}
		}
		else
		{
			//transmission
			float distance = glm::length(hitPosition - origin);
			float inMediaAttenuation = 1.0f;
			/*
			if (glm::dot(direction, hitNormal) >= 0.0f) // inside
			{
				inMediaAttenuation = 1 / (1 + 0.2 * distance);
			}*/

			if (NEESwitch) // NEE is on
			{
				glm::vec3 tempColor = directIntensity(direction, hitPosition, hitNormal, hitMaterial);
				outputColor += tempColor;
			}
			glm::vec3 tempColor = nextIntensity(direction, hitPosition, hitNormal,
				hitMaterial, inMediaAttenuation * throughPut, NEESwitch, depth);
			outputColor += inMediaAttenuation * tempColor;
		}

	}
	else
	{
		//TO DO: environment mapping
	}

	return outputColor;
}

void PathTracerIntegrator::sampleHemisphere(glm::vec3 normal,
	std::vector<glm::vec3>* sampleDirect)
{
	glm::vec3 w = glm::normalize(normal);
	glm::vec3 u = glm::cross(w, glm::vec3(0,1,0));

	if (glm::length(u) < 0.0001)
	{
		u = glm::cross(w, glm::vec3(1, 0, 0));
	}

	u = glm::normalize(u);
	glm::vec3 v = glm::cross(w, u);

	float theta = glm::acos(getRandNum());
	float phi = 2 * PI * getRandNum();

	glm::vec3 tempDir = glm::cos(phi) * glm::sin(theta) * u;
	tempDir += glm::sin(phi) * glm::sin(theta) * v;
	tempDir += glm::cos(theta) * w;

	sampleDirect->push_back(tempDir);
}

void PathTracerIntegrator::sampleCosine(glm::vec3 normal,
	std::vector<glm::vec3>* sampleDirect)
{
	glm::vec3 w = glm::normalize(normal);
	glm::vec3 u = glm::cross(w, glm::vec3(0, 1, 0));

	if (glm::length(u) < 0.0001)
	{
		u = glm::cross(w, glm::vec3(1, 0, 0));
	}

	u = glm::normalize(u);
	glm::vec3 v = glm::cross(w, u);

	float theta = glm::acos(sqrt(getRandNum()));
	float phi = 2 * PI * getRandNum();

	glm::vec3 tempDir = glm::cos(phi) * glm::sin(theta) * u;
	tempDir += glm::sin(phi) * glm::sin(theta) * v;
	tempDir += glm::cos(theta) * w;

	sampleDirect->push_back(tempDir);
}

void PathTracerIntegrator::sampleQuadLight(glm::vec3 origin, quadLight_t light,
	std::vector<glm::vec3>* sampleDirect, std::vector<float>* sampleLength)
{

	glm::vec3 lightNorm = glm::cross(light.ab, light.ac);

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
				lightRayDir = lightRayDir / lightRayLen;

				if (glm::dot(lightRayDir, lightNorm) > 0)
				{
					sampleDirect->push_back(lightRayDir);
					sampleLength->push_back(lightRayLen);
				}
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
			lightRayDir = lightRayDir / lightRayLen;

			if (glm::dot(lightRayDir, lightNorm) > 0)
			{
				sampleDirect->push_back(lightRayDir);
				sampleLength->push_back(lightRayLen);
			}
		}
	}
}


glm::vec3 PathTracerIntegrator::directIntensity(glm::vec3 direction, glm::vec3 hitPosition, 
	glm::vec3 hitNormal, Material* hitMaterial)
{
	glm::vec3 outputColor = glm::vec3(0.0f, 0.0f, 0.0f);

	for(size_t j = 0; j < _scene->quadLights.size(); j++) // each light
	{
		quadLight_t light = _scene->quadLights[j];
		glm::vec3 lightNorm = glm::cross(light.ab, light.ac);
		float quadLightArea = glm::length(lightNorm);
		lightNorm /= quadLightArea;

		std::vector<glm::vec3> sampleDirect;
		std::vector<float> sampleLength;
		sampleQuadLight(hitPosition, light, &sampleDirect, &sampleLength);

		glm::vec3 sumDirectSample(0.0f);
		for (size_t i = 0; i < sampleDirect.size(); i++) // each sample
		{
			bool occluded =
				_scene->castOcclusionRay(hitPosition, sampleDirect[i], sampleLength[i]);
			if (!occluded)
			{
				glm::vec3 f
					= hitMaterial->getBSDF(hitNormal, sampleDirect[i], -direction);

				float cosTheta_i = glm::dot(hitNormal, sampleDirect[i]);
				cosTheta_i = std::max(cosTheta_i, 0.0f);
				if (_scene->NEEFlag != 2)
				{
					float cosTheta_o = glm::dot(lightNorm, sampleDirect[i]);
					cosTheta_o = std::max(cosTheta_o, 0.0f);

					sumDirectSample += f * cosTheta_i * cosTheta_o
						/ (sampleLength[i] * sampleLength[i]) * quadLightArea;
				}
				else
				{
					double lightPDF = pdfLight(j, sampleDirect[i], sampleLength[i]);
					double brdfPDF
						= hitMaterial->getPDF(hitNormal, sampleDirect[i], -direction);
					double weight = pow(lightPDF, 2) / (pow(brdfPDF, 2) + pow(lightPDF, 2));
					
					sumDirectSample += f *  (float)(cosTheta_i / lightPDF * weight);
				}
				
			}
		}

		glm::vec3 L_d = light.intensity * sumDirectSample / (float)_scene->lightsamples;
		outputColor += L_d;
	}
	if (_scene->NEEFlag == 2)
	{
		outputColor /= (float)_scene->quadLights.size();
	}


	return outputColor;
}

glm::vec3 PathTracerIntegrator::nextIntensity(glm::vec3 direction, glm::vec3 hitPosition,
	glm::vec3 hitNormal, Material* hitMaterial, glm::vec3 throughPut, bool NEESwitch, int depth)
{
	float p_termin = 0.0f;
	if (_scene->RRFlag) //RUSSIAN ROULETTE
	{
		p_termin = std::max(throughPut.r, throughPut.g);
		p_termin = std::max(p_termin, throughPut.b);
		p_termin = std::min(p_termin, 1.0f);
		p_termin = std::max(p_termin, 0.0f);
		p_termin = 1 - p_termin;

		if (p_termin > getRandNum())
		{
			return glm::vec3(0);
		}
	}


	std::vector<glm::vec3> sampleNextDirect;
	glm::vec3 indirectColor(0.0f);

	if (_scene->importanceSampling == 1) // hemisphere
	{
		sampleHemisphere(hitNormal, &sampleNextDirect);

		for (size_t i = 0; i < sampleNextDirect.size(); i++) // each sample
		{
			//BRDF
			glm::vec3 f_Phong = hitMaterial->getBSDF(hitNormal, sampleNextDirect[i], -direction);

			//cos term
			float cosTheta_i = glm::dot(hitNormal, sampleNextDirect[i]);
			cosTheta_i = std::max(cosTheta_i, 0.0f);

			//throughput
			glm::vec3 nextThroughput = 2 * PI * f_Phong * cosTheta_i * throughPut / (1 - p_termin);

			//the color from next path
			int hitLightID = -1;
			glm::vec3 tempColor = traceRay(hitPosition, sampleNextDirect[i], depth - 1,
				nextThroughput, NEESwitch, &hitLightID);
			if ((hitLightID != -1) && NEESwitch) continue;

			indirectColor += 2 * PI * f_Phong * cosTheta_i * tempColor / (1 - p_termin); // N=1, only 1 sample
		}
	}
	else if (_scene->importanceSampling == 2) // cosine
	{
		sampleCosine(hitNormal, &sampleNextDirect);

		//glm::vec3 sumIndirectSample(0.0f);
		for (size_t i = 0; i < sampleNextDirect.size(); i++) // each sample
		{
			//BRDF
			glm::vec3 f_Phong = hitMaterial->getBSDF(hitNormal, sampleNextDirect[i], -direction);

			//throughput
			glm::vec3 nextThroughput = PI * f_Phong * throughPut / (1 - p_termin);

			//the color from next path
			int hitLightID = -1;
			glm::vec3 tempColor = traceRay(hitPosition, sampleNextDirect[i], depth - 1,
				nextThroughput, NEESwitch, &hitLightID);
			if ((hitLightID != -1) && NEESwitch) continue;

			indirectColor += PI * f_Phong * tempColor / (1 - p_termin); // N=1, only 1 sample
		}
	}
	else if (_scene->importanceSampling == 3)// BRDF
	{
		hitMaterial->sampleBSDF(hitNormal, -direction, &sampleNextDirect);

		for (size_t i = 0; i < sampleNextDirect.size(); i++) // each sample
		{
			float temp = glm::dot(sampleNextDirect[i], hitNormal);
			//if (glm::dot(sampleNextDirect[i], hitNormal) > 0.0f) // above visible hemisphere
			{
				double pdfBRDF;
				glm::vec3 f = hitMaterial->getBSDF(hitNormal, sampleNextDirect[i], -direction, &pdfBRDF);

				//cos term
				//float cosTheta_i = glm::dot(hitNormal, sampleNextDirect[i]);
				float cosTheta_i = abs(glm::dot(hitNormal, sampleNextDirect[i]));

				//throughput
				glm::vec3 nextThroughput = f * throughPut * (float)(cosTheta_i
					/ (1 - p_termin) / pdfBRDF);

				if (std::isnan(nextThroughput.r))
				{
					printf("Throughput nan\n");
				}

				//the color from next path
				int hitLightID = -1;
				double lightDis;
				glm::vec3 tempColor = traceRay(hitPosition, sampleNextDirect[i], depth - 1,
					nextThroughput, NEESwitch, &hitLightID, &lightDis);
				if ((hitLightID != -1) && NEESwitch) continue;

				double weight = 1.0;
				if (!NEESwitch && (_scene->NEEFlag == 2))//MIS
				{
					double pdfNEE = pdfLight(hitLightID, sampleNextDirect[i], lightDis);
					weight = pow(pdfBRDF, 2) / (pow(pdfNEE, 2) + pow(pdfBRDF, 2));
				}

				indirectColor += f * tempColor * (float)(cosTheta_i
					/ (1 - p_termin) / pdfBRDF * weight); // N=1, only 1 sample

				if (std::isnan(indirectColor.r))
				{
					printf("next intensity nan\n");
				}
			}
		}
	}

	return indirectColor;
}


double PathTracerIntegrator::pdfLight(int lightID, glm::vec3 omega_i, float distance)
{
	if (lightID == -1)
	{
		return 0.0;
	}

	quadLight_t light = _scene->quadLights.at(lightID);
	glm::vec3 lightNorm = glm::cross(light.ab, light.ac);
	float quadLightArea = glm::length(lightNorm);
	lightNorm /= quadLightArea;

	double cosTheta_ni = abs(glm::dot(lightNorm, omega_i));
	double pdf = pow((double)distance, 2) / (double)quadLightArea / cosTheta_ni;

	return pdf / (double)_scene->quadLights.size();
}