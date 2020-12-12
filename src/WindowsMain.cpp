#include <iostream>
#include <stdexcept>

#include <Windows.h>

#include "Engine.h"

int main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    try {

        char filePath[MAX_PATH];

        OPENFILENAME ofn;
        ZeroMemory(&filePath, sizeof(filePath));
        ZeroMemory(&ofn, sizeof(ofn));
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = nullptr;
        ofn.lpstrFilter = "Scene Files\0*.test\0";
        ofn.lpstrFile = filePath;
        ofn.nMaxFile = MAX_PATH;
        ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

        if (!GetOpenFileNameA(&ofn)) {
            throw std::runtime_error("No scene file.");
        }

        render(filePath);

    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
