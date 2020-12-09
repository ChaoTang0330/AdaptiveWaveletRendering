#include <iostream>
#include <stdexcept>

#include "Engine.h"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: ./rt168 <scene file>" << std::endl;
        return 1;
    }

    std::string sceneFilePath = argv[1];

    try {
        render(sceneFilePath);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
