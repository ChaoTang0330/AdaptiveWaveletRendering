#pragma once
#include <thread>

const float PI_OVER_FOUR = 0.78539816339f;
const float PI_OVER_TWO = 1.57079632679f;
const float PI = 3.14159265359f;
const float TWO_PI = 6.28318530718f;
const float FOUR_PI = 12.5663706144f;
const float INV_PI = 0.31830988618f;
const float INV_TWO_PI = 0.15915494309f;
const float INV_FOUR_PI = 0.07957747154f;

const float EPSILON = 1e-5f;

#define RAND_MULTIPLIER 6364136223846793005ULL
#define RNAD_INCREMENT 1442695040888963407ULL

static float getRandNum()
{
	thread_local unsigned long long seed = std::hash<std::thread::id>{}(std::this_thread::get_id());
	seed = (RAND_MULTIPLIER * seed) + RNAD_INCREMENT;
	return static_cast<float>(seed & 0xffffffff) / 0xffffffff;
}
