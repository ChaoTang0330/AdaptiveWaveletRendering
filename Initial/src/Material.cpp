#include "Material.h"

bool Material::getRefractIdx(float* idx)
{
	if (idx != NULL)
	{
		*idx = refractIdx;
	}

	if (refractIdx != 0.0f)
	{
		return true;
	}
	return false;
}

glm::vec3 Material::getRefractDir(glm::vec3 inDir, glm::vec3 normal, float* nextRefractIdx) // assume oneside is air
{
	if (refractIdx > 0.0f)
	{
		float eta = 1.0f; 
		normal = glm::normalize(normal);
		inDir = glm::normalize(inDir);
		if (glm::dot(-inDir, normal) <= 0.0f) // in and normal not in same side -> inside media
		{
			normal = -normal; //->flip normal
			eta = refractIdx;
			*nextRefractIdx = 1.0; //refract to air
		}
		else
		{
			eta = 1 / refractIdx;
			*nextRefractIdx = refractIdx; //refract to media
		}

		glm::vec3 result = glm::refract(inDir, normal, eta);

		return glm::normalize(result);
	}
	else
	{
		*nextRefractIdx = 1.0;
		return glm::vec3(0.0);
	}
	
}

float Material::getTransRate(glm::vec3 inDir, glm::vec3 transDir, glm::vec3 normal)
{
	float nextRefIdx = refractIdx;
	float currRefIdx = 1.0;
	if (glm::dot(-inDir, normal) <= 0.0f) //inside
	{
		normal = -normal;
		currRefIdx = refractIdx;
		nextRefIdx = 1.0f;
	}
	//Fresnel
	float cosTheta_i = glm::dot(-inDir, normal);
	float cosTheta_t = glm::dot(transDir, -normal);

	float R_s = (currRefIdx * cosTheta_i - nextRefIdx * cosTheta_t) 
		/ (currRefIdx * cosTheta_i + nextRefIdx * cosTheta_t);
	R_s = pow(R_s, 2);

	float R_p = (currRefIdx * cosTheta_t - nextRefIdx * cosTheta_i) 
		/ (currRefIdx * cosTheta_t + nextRefIdx * cosTheta_i);
	R_p = pow(R_p, 2);

	return 1 - (R_s + R_p) / 2.0f;
}

/**********************Phong**********************/
PhongMaterial::PhongMaterial(material_t currMaterial)
{
    diffuse = currMaterial.diffuse;
    specular = currMaterial.specular;
    shininess = currMaterial.shininess;
    emission = currMaterial.emission;
    ambient = currMaterial.ambient;

    //transmission
    refractIdx = currMaterial.refractIdx;

	specular_avg =
		(specular.r + specular.g + specular.b) / 3.0f;
	diffuse_avg =
		(diffuse.r + diffuse.g + diffuse.b) / 3.0f;
	t = specular_avg / (diffuse_avg + specular_avg);
}

glm::vec3  PhongMaterial::getBSDF(glm::vec3 normal, glm::vec3 omega_i, glm::vec3 omega_o, double* pdfBRDF)
{
	normal = glm::normalize(normal);
	omega_i = glm::normalize(omega_i);
	omega_o = glm::normalize(omega_o);

	glm::vec3 reflection = glm::reflect(-omega_o, normal);
	reflection = glm::normalize(reflection);
	//phone BRDF
	double specularFactor = glm::dot(reflection, omega_i);
	specularFactor = std::max(specularFactor, 0.0);

	double factor = (0.5 * shininess + 1.0) *
		pow(specularFactor, (double)shininess);

	if (std::isnan(factor))
	{
		printf("nan BRDF phong\n");
	}

	glm::vec3 f_Phong = specular * (float)factor;
	f_Phong += diffuse;
	f_Phong /= PI;

	if (pdfBRDF != NULL)
	{
		//pdf
		*pdfBRDF = (1 - t) * glm::dot(normal, omega_i) / PI
			+ t * (shininess + 1.0)
			* pow(specularFactor, shininess)
			/ (2.0 * PI);
	}

	if (std::isnan(f_Phong.r))
	{
		printf("nan BRDF phong\n");
	}

	return f_Phong;
}

double PhongMaterial::getPDF(glm::vec3 normal, glm::vec3 omega_i, glm::vec3 omega_o)
{
	glm::vec3 reflection = glm::reflect(-omega_o, normal);
	//phone BRDF
	double specularFactor = glm::dot(reflection, omega_i);
	specularFactor = std::max(specularFactor, 0.0);

	

	double pdf = (1 - t) * glm::dot(normal, omega_i) / PI
		+ t * ((double)shininess + 1.0)
		* pow(specularFactor, shininess)
		/ (2.0 * (double)PI);

	return pdf;
}

void PhongMaterial::sampleBSDF(glm::vec3 normal, glm::vec3 omega_o, std::vector<glm::vec3>* sampleDirect)
{
	glm::vec3 w;
	float theta;

	glm::vec3 reflection = glm::reflect(-omega_o, normal);
	if (getRandNum() <= t) // specular
	{
		theta = glm::acos(pow(getRandNum(), 1.0f / (shininess + 1.0f)));
		w = glm::normalize(reflection);
	}
	else // diffuse
	{
		theta = glm::acos(sqrt(getRandNum()));
		w = glm::normalize(normal);
	}


	glm::vec3 u = glm::cross(w, glm::vec3(0, 1, 0));

	if (glm::length(u) < 0.0001)
	{
		u = glm::cross(w, glm::vec3(1, 0, 0));
	}

	u = glm::normalize(u);
	glm::vec3 v = glm::cross(w, u);

	float phi = 2 * PI * getRandNum();

	glm::vec3 tempDir = glm::cos(phi) * glm::sin(theta) * u;
	tempDir += glm::sin(phi) * glm::sin(theta) * v;
	tempDir += glm::cos(theta) * w;

	tempDir = glm::normalize(tempDir);
	if (glm::dot(tempDir, normal) > 0.0)
	{
		sampleDirect->push_back(tempDir); // inside visible hemisphere
	}
}

/*******************GGX*************************************/
GGXMaterial::GGXMaterial(material_t currMaterial)
{
	diffuse = currMaterial.diffuse;
	specular = currMaterial.specular;
	shininess = currMaterial.shininess;
	emission = currMaterial.emission;
	ambient = currMaterial.ambient;

	//GGX
	roughness = currMaterial.roughness;

	//transmission
	refractIdx = currMaterial.refractIdx;

	specular_avg =
		(specular.r + specular.g + specular.b) / 3.0f;
	diffuse_avg =
		(diffuse.r + diffuse.g + diffuse.b) / 3.0f;
	t = specular_avg / (diffuse_avg + specular_avg);
	t = std::max(0.25, t);
}

glm::vec3 GGXMaterial::getBSDF(glm::vec3 normal, glm::vec3 omega_i, glm::vec3 omega_o, double* pdfBRDF)
{
	normal = glm::normalize(normal);
	omega_i = glm::normalize(omega_i);
	omega_o = glm::normalize(omega_o);

	//BRDF
	glm::vec3 h = glm::normalize(omega_i + omega_o);
	normal = glm::normalize(normal);
	double cosTheta_hn = glm::dot(h, normal);
	double cosTheta_hn2 = pow(cosTheta_hn, 2);

	double alpha_2 = pow(roughness, 2);

	double D_h = PI * pow((alpha_2 - 1) * cosTheta_hn2 + 1, 2);
	D_h = alpha_2 / D_h;

	double cosTheta_ni = glm::dot(omega_i, normal);
	double cosTheta_no = glm::dot(omega_o, normal);

	if (cosTheta_ni <= 0.0 || cosTheta_no <= 0.0)
	{
		if (pdfBRDF != NULL)
		{
			*pdfBRDF = 1.0;
		}
		return glm::vec3(0.0f);
	}


	double G1_i = 2.0f / (1 + sqrt(1 + alpha_2 * (pow(cosTheta_ni, -2) - 1)));
	double G1_o = 2.0f / (1 + sqrt(1 + alpha_2 * (pow(cosTheta_no, -2) - 1)));

	double cosTheta_hi = glm::dot(omega_i, h);

	glm::vec3 F = specular
		+ (glm::vec3(1.0) - specular)
		* (float)pow(1.0 - cosTheta_hi, 5);

	glm::vec3 f_GGX = diffuse / PI
		+ F * (float)(G1_i * G1_o * D_h / (4 * cosTheta_ni * cosTheta_no));


	if (pdfBRDF != NULL)
	{
		*pdfBRDF = (1 - t) * cosTheta_ni / PI
			+ t * D_h * cosTheta_hn / (4 * cosTheta_hi);
	}

	return f_GGX;
}

double GGXMaterial::getPDF(glm::vec3 normal, glm::vec3 omega_i, glm::vec3 omega_o)
{
	normal = glm::normalize(normal);
	omega_i = glm::normalize(omega_i);
	omega_o = glm::normalize(omega_o);

	glm::vec3 h = glm::normalize(omega_i + omega_o);
	normal = glm::normalize(normal);
	double cosTheta_hn = glm::dot(h, normal);
	double cosTheta_hn2 = pow(cosTheta_hn, 2);

	double alpha_2 = pow(roughness, 2);

	double D_h = PI * pow((alpha_2 - 1) * cosTheta_hn2 + 1, 2);
	D_h = alpha_2 / D_h;

	double cosTheta_ni = glm::dot(omega_i, normal);
	double cosTheta_no = glm::dot(omega_o, normal);

	double cosTheta_hi = glm::dot(omega_i, h);

	//pdf
	double pdf = (1 - t) * cosTheta_ni / PI
		+ t * D_h * cosTheta_hn / (4 * cosTheta_hi);

	return pdf;
}

void GGXMaterial::sampleBSDF(glm::vec3 normal, glm::vec3 omega_o, std::vector<glm::vec3>* sampleDirect)
{
	normal = glm::normalize(normal);
	omega_o = glm::normalize(omega_o);

	float theta;
	glm::vec3 w = glm::normalize(normal);

	glm::vec3 u = glm::cross(w, glm::vec3(0, 1, 0));
	glm::vec3 u_1 = glm::cross(w, glm::vec3(1, 0, 0));
	if (glm::length(u) < glm::length(u_1))
	{
		u = u_1;
	}
	u = glm::normalize(u);

	glm::vec3 v = glm::cross(w, u);

	float phi = 2 * PI * getRandNum();

	glm::vec3 sample;

	if (getRandNum() <= t) // specular
	{
		float tempRandNum = getRandNum();
		float tanTheta = roughness * sqrt(tempRandNum) / sqrt(1 - tempRandNum);
		theta = glm::atan(tanTheta);

		glm::vec3 half = glm::cos(phi) * glm::sin(theta) * u;
		half += glm::sin(phi) * glm::sin(theta) * v;
		half += glm::cos(theta) * w;

		sample = glm::reflect(-omega_o, half);
	}
	else // diffuse
	{
		theta = glm::acos(sqrt(getRandNum()));

		sample = glm::cos(phi) * glm::sin(theta) * u;
		sample += glm::sin(phi) * glm::sin(theta) * v;
		sample += glm::cos(theta) * w;
	}

	sample = glm::normalize(sample);
	if (glm::dot(sample, normal) > 0.0)
	{
		sampleDirect->push_back(sample); // inside visible hemisphere
	}
}


/***************************GGXTrans**************************/
GGXTransMaterial::GGXTransMaterial(material_t currMaterial)
{
	diffuse = currMaterial.diffuse;
	specular = currMaterial.specular;
	shininess = currMaterial.shininess;
	emission = currMaterial.emission;
	ambient = currMaterial.ambient;

	//GGX
	roughness = currMaterial.roughness;
	//transmission
	refractIdx = currMaterial.refractIdx;
}


glm::vec3 GGXTransMaterial::getBSDF(glm::vec3 normal, glm::vec3 omega_i, glm::vec3 omega_o, double* pdfBSDF)
{
	glm::vec3 BSDF;

	normal = glm::normalize(normal);
	omega_i = glm::normalize(omega_i);
	omega_o = glm::normalize(omega_o);

	bool reflectionFlag, insideFlag;
	glm::vec3 half;
	float eta_i, eta_o;

	if (glm::dot(normal, omega_i) >= 0.0f) //outside
	{
		eta_i = 1.0f;
		eta_o = refractIdx;
		insideFlag = false;
	}
	else // inside
	{
		eta_o = 1.0f;
		eta_i = refractIdx;
		//normal = -normal;
		insideFlag = true;
	}

	if ((double)glm::dot(normal, omega_o) * (double)glm::dot(normal, omega_i) >= 0.0) // reflection
	{
		float sign = (glm::dot(omega_i, normal) > 0) ? 1.0 : -1.0;
		half = sign * (omega_i + omega_o);
		half = glm::normalize(half);
		reflectionFlag = true;
	}
	else // refraction
	{
		half = -(eta_i * omega_i + eta_o * omega_o);
		half = glm::normalize(half);
		reflectionFlag = false;
	}

	//Fresnel Term
	double F, g, c;
	c = abs(glm::dot(omega_i, half));
	g = pow(eta_o / eta_i, 2) - 1.0 + pow(c, 2);
	if (g > 0.0)
	{
		g = sqrt(g);
		F = 0.5 * pow((g - c) / (g + c), 2) 
			* (1 + pow((c * (g + c) - 1) / (c * (g - c) + 1), 2));
	}
	else
	{
		F = 1.0;
	}
	
	//Distribution
	double D;
	double cosTheta_hn = glm::dot(half, normal);
	if (cosTheta_hn <= 0.0)
	{
		D = 0.0;
		/*
		double cosTheta_hn2 = pow(cosTheta_hn, 2);
		double alpha2 = pow(roughness, 2);
		D = PI * pow((alpha2 - 1) * cosTheta_hn2 + 1, 2);
		D = pow(roughness, 2) / D;*/
	}
	else
	{
		double cosTheta_hn2 = pow(cosTheta_hn, 2);
		double alpha2 = pow(roughness, 2);
		D = PI * pow((alpha2 - 1) * cosTheta_hn2 + 1, 2);
		D = pow(roughness, 2) / D;
	}

	//G, Smith term
	double G, G1, G2;

	double cosTheta_hi = glm::dot(half, omega_i);
	double cosTheta_ni = glm::dot(normal, omega_i);
	if (cosTheta_hi * cosTheta_ni >= -0.001) // change to multiply
	{
		double tanTheta_ni2 = 1.0 / pow(cosTheta_ni, 2) - 1.0;
		G1 = 1.0 + sqrt(1 + pow(roughness, 2) * tanTheta_ni2);
		G1 = 2.0 / G1;
	}
	else
	{
		G1 = 0.0;

		//test
		/*
		double tanTheta_ni2 = 1.0 / pow(cosTheta_ni, 2) - 1.0;
		G1 = 1.0 + sqrt(1 + pow(roughness, 2) * tanTheta_ni2);
		G1 = 2.0 / G1;*/
	}
	
	double cosTheta_ho = glm::dot(half, omega_o);
	double cosTheta_no = glm::dot(normal, omega_o);
	if (cosTheta_ho * cosTheta_no >= -0.001) // change to multiply
	{
		double tanTheta_no2 = 1 / pow(cosTheta_no, 2) - 1.0;
		G2 = 1.0 + sqrt(1 + pow(roughness, 2) * tanTheta_no2);
		G2 = 2.0 / G2;
	}
	else
	{
		G2 = 0.0;

		//test
		/*
		double tanTheta_no2 = 1 / pow(cosTheta_no, 2) - 1.0;
		G2 = 1.0 + sqrt(1 + pow(roughness, 2) * tanTheta_no2);
		G2 = 2.0 / G2;*/
	}

	G = G1 * G2;


	//BSDF
	if (reflectionFlag) // reflection
	{
		double result = F * G * D / (4 * abs(cosTheta_ni) * abs(cosTheta_ho));
		if (std::isnan(result))
		{
			printf("nan reflect getBSDF\n");
			result = 0.0f;
		}
		BSDF = glm::vec3((float)result, (float)result, (float)result);
	}
	else
	{
		double result = abs(cosTheta_hi * cosTheta_ho / cosTheta_ni / cosTheta_no);
		result *= pow(eta_o, 2) * (1 - F) * G * D;
		result /= pow(eta_i * cosTheta_hi + eta_o * cosTheta_ho, 2);
		if (std::isnan(result))
		{
			printf("nan refract getBSDF\n");
		}
		BSDF = glm::vec3((float)result, (float)result, (float)result);
	}


	if (pdfBSDF != NULL)
	{
		*pdfBSDF = D * abs(cosTheta_hn);
		double diff;

		if (reflectionFlag)
		{
			diff = 0.25 / abs(cosTheta_ho);
		}
		else
		{
			diff = pow(eta_o, 2) * abs(cosTheta_ho)
				/ pow(eta_i * cosTheta_hi + eta_o * cosTheta_ho, 2);
		}

		*pdfBSDF *= diff;

		if (*pdfBSDF == 0.0 || std::isnan(*pdfBSDF))
		{
			*pdfBSDF = 1.0;
		}
	}

	return BSDF;
}

double GGXTransMaterial::getPDF(glm::vec3 normal, glm::vec3 omega_i, glm::vec3 omega_o)
{
	normal = glm::normalize(normal);
	omega_i = glm::normalize(omega_i);
	omega_o = glm::normalize(omega_o);

	double pdfBSDF;

	bool reflectionFlag;
	glm::vec3 half;
	float eta_i, eta_o;

	if (glm::dot(normal, omega_i) > 0.0f) //outside
	{
		eta_i = 1.0f;
		eta_o = refractIdx;
	}
	else // inside
	{
		eta_o = 1.0f;
		eta_i = refractIdx;
	}

	if ((double)glm::dot(normal, omega_o) * (double)glm::dot(normal, omega_i) >= 0.0) // reflection
	{
		float sign = (glm::dot(omega_i, normal) > 0) ? 1.0 : -1.0;
		half = sign * (omega_i + omega_o);
		half = glm::normalize(half);
		reflectionFlag = true;
	}
	else // refraction
	{
		half = -(eta_i * omega_i + eta_o * omega_o);
		half = glm::normalize(half);
		reflectionFlag = false;
	}

	//Distribution
	double D;
	double cosTheta_hn = glm::dot(half, normal);
	if (cosTheta_hn <= 0.0)
	{
		D = 0.0;
	}
	else
	{
		double cosTheta_hn2 = pow(cosTheta_hn, 2);
		double alpha2 = pow(roughness, 2);
		D = PI * pow((alpha2 - 1) * cosTheta_hn2 + 1, 2);
		D = pow(roughness, 2) / D;
	}

	double cosTheta_hi = glm::dot(half, omega_i);
	double cosTheta_ni = glm::dot(normal, omega_i);
	double cosTheta_ho = glm::dot(half, omega_o);
	double cosTheta_no = glm::dot(normal, omega_o);

	pdfBSDF = D * abs(cosTheta_hn);
	double diff;

	if (reflectionFlag)
	{
		diff = 0.25 / abs(cosTheta_ho);
	}
	else
	{
		diff = pow(eta_o, 2) * abs(cosTheta_ho)
			/ pow(eta_i * cosTheta_hi + eta_o * cosTheta_ho, 2);
	}

	pdfBSDF *= diff;


	if (pdfBSDF == 0.0 ||std::isnan(pdfBSDF))
	{
		pdfBSDF = 1.0;
	}

	return pdfBSDF;
}

void GGXTransMaterial::sampleBSDF(glm::vec3 normal, glm::vec3 omega_i, std::vector<glm::vec3>* sampleDirect)
{
	normal = glm::normalize(normal);
	omega_i = glm::normalize(omega_i);

	/****test*****/
	//glm::vec3 currNormal = normal;

	glm::vec3 half;
	float eta_i, eta_o;

	if (glm::dot(normal, omega_i) >= 0.0f) //outside
	{
		eta_i = 1.0f;
		eta_o = refractIdx;
	}
	else // inside
	{
		eta_o = 1.0f;
		eta_i = refractIdx;
	}

	glm::vec3 w = glm::normalize(normal);

	glm::vec3 u = glm::cross(w, glm::vec3(0, 1, 0));
	glm::vec3 u_1 = glm::cross(w, glm::vec3(1, 0, 0));
	if (glm::length(u) < glm::length(u_1))
	{
		u = u_1;
	}
	u = glm::normalize(u);

	glm::vec3 v = glm::cross(w, u);

	double tempRand1 = getRandNum();
	float tanTheta = roughness * sqrt(tempRand1) / sqrt(1 - tempRand1);
	float theta = glm::atan(tanTheta);
	float phi = 2 * PI * getRandNum();

	half = glm::cos(phi) * glm::sin(theta) * u;
	half += glm::sin(phi) * glm::sin(theta) * v;
	half += glm::cos(theta) * w;
	half = glm::normalize(half);


	//Fresnel Term
	double F, g, c;
	c = abs(glm::dot(omega_i, half));
	g = pow(eta_o / eta_i, 2) - 1.0 + pow(c, 2);
	if (g > 0.0)
	{
		g = sqrt(g);
		F = 0.5 * pow((g - c) / (g + c), 2)
			* (1 + pow((c * (g + c) - 1) / (c * (g - c) + 1), 2));
	}
	else
	{
		F = 1.0;
	}

	
	if (getRandNum() < (float)F) // reflect
	{
		double cosTheta_hi = glm::dot(omega_i, half);
		glm::vec3 tempDir = (float)(2 * cosTheta_hi) * half - omega_i;
		
		tempDir = glm::normalize(tempDir);

		if ((double)glm::dot(tempDir, normal) * (double)glm::dot(omega_i, normal) >= 0.0)
		{
			sampleDirect->push_back(tempDir); // inside visible hemisphere
		}
	}
	else // refract
	{
		double eta = eta_i / eta_o;

		glm::vec3 tempDir;
		if (glm::dot(half, omega_i) >= 0.0f)
		{
			tempDir = glm::refract(-omega_i, half, (float)eta);
		}
		else
		{
			tempDir = glm::refract(-omega_i, -half, (float)eta);
		}

		if ((double)glm::dot(tempDir, normal) * (double)glm::dot(omega_i, normal) <= 0.0)
		{
			tempDir = glm::normalize(tempDir);
			sampleDirect->push_back(tempDir); // inside visible hemisphere
		}
	}
}