#pragma once

#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <algorithm>

#include "Constants.h"

struct material_t {
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
    glm::vec3 emission;
    glm::vec3 ambient;

    //HW4
    float roughness = 0.0f;
    int BRDFtype = 0;

    //transmission
    float refractIdx = 0.0f;
};

class Material
{
protected:
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
    glm::vec3 emission;
    glm::vec3 ambient;

    //transmission
    float refractIdx = 0.0f;
    
public:
    Material() {};

    virtual glm::vec3 getBSDF(glm::vec3 normal, glm::vec3 omega_i, glm::vec3 omega_o, double* pdfBRDF = NULL) = 0;
    virtual double getPDF(glm::vec3 normal, glm::vec3 omega_i, glm::vec3 omega_o) = 0;
    virtual void sampleBSDF(glm::vec3 normal, glm::vec3 omega_o, std::vector<glm::vec3>* sampleDirect) = 0;

    bool getRefractIdx(float* idx = NULL);
    glm::vec3 getRefractDir(glm::vec3 inDir, glm::vec3 normal, float *nextRefractIdx);
    float getTransRate(glm::vec3 inDir, glm::vec3 transDir, glm::vec3 normal);

    
};

class PhongMaterial : public Material
{
private:
    double specular_avg;
    double diffuse_avg;
    double t;

public:
    PhongMaterial(material_t currMaterial);

    glm::vec3 getBSDF(glm::vec3 normal, glm::vec3 omega_i, glm::vec3 omega_o, double* pdfBRDF = NULL);
    double getPDF(glm::vec3 normal, glm::vec3 omega_i, glm::vec3 omega_o);
    void sampleBSDF(glm::vec3 normal, glm::vec3 omega_o, std::vector<glm::vec3>* sampleDirect);
};

class GGXMaterial : public Material
{
private:
    float roughness = 0.0f;

    double specular_avg;
    double diffuse_avg;
    double t;

public:
    GGXMaterial(material_t currMaterial);

    glm::vec3 getBSDF(glm::vec3 normal, glm::vec3 omega_i, glm::vec3 omega_o,double* pdfBRDF = NULL);
    double getPDF(glm::vec3 normal, glm::vec3 omega_i, glm::vec3 omega_o);
    void sampleBSDF(glm::vec3 normal, glm::vec3 omega_o, std::vector<glm::vec3>* sampleDirect);
};

class GGXTransMaterial : public Material
{
private:
    float roughness = 0.0f;

public:
    GGXTransMaterial(material_t currMaterial);

    glm::vec3 getBSDF(glm::vec3 normal, glm::vec3 omega_i, glm::vec3 omega_o, double* pdfBSDF = NULL);
    double getPDF(glm::vec3 normal, glm::vec3 omega_i, glm::vec3 omega_o);
    void sampleBSDF(glm::vec3 normal, glm::vec3 omega_o, std::vector<glm::vec3>* sampleDirect);
};