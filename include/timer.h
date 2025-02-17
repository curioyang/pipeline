#include <chrono>
#include <string>
#include <iostream>

class Timer {
public:
    Timer(const std::string& name) : name(name), start_time(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        std::cout << name << " took " << duration / 1e6 << " ms." << std::endl;
    }

private:
    std::string name;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};
