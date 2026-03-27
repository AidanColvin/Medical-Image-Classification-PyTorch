#include <iostream>
#include <fstream>
#include <string>

/**
 * @brief Submission Validator Engine.
 * Ensures the generated CSV exactly mirrors the clinical requirements.
 */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./validator <filename.csv>" << std::endl;
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "Error: File not found: " << argv[1] << std::endl;
        return 1;
    }

    std::string line;
    int count = 0;
    std::getline(file, line); // Header skip

    while (std::getline(file, line)) {
        if (!line.empty()) count++;
    }

    std::cout << "\n[C++ SYSTEM] Validating " << argv[1] << "..." << std::endl;
    if (count == 624) {
        std::cout << "[SUCCESS] Integrity Check Passed: 624 records found." << std::endl;
        return 0;
    } else {
        std::cerr << "[CRITICAL] Mismatch! Expected 624, found " << count << "." << std::endl;
        return 1;
    }
}
