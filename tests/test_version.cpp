#include <cassert>
#include <iostream>
#include "keryth/version.hpp"

int main() {
    assert(keryth::VERSION_MAJOR == 0);
    assert(keryth::VERSION_MINOR == 1);
    assert(keryth::VERSION_PATCH == 0);
    assert(keryth::get_version() == "0.1.0");
    assert(keryth::get_tagline() == "Write once. Optimize for the machine.");

    std::cout << "test_version passed successfully.\n";
    return 0;
}
