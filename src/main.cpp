#include <iostream>
#include <string_view>
#include "keryth/version.hpp"

namespace {

void print_version() {
    std::cout << "Keryth Compiler v" << keryth::get_version() << "\n"
              << keryth::get_tagline() << "\n";
}

void print_help(std::string_view program_name) {
    std::cout << "Usage: " << program_name << " [options] [source_file.kr]\n\n"
              << "Keryth: An AI-native programming language and optimizing compiler.\n\n"
              << "Options:\n"
              << "  -h, --help       Show this help message and exit\n"
              << "  -v, --version    Show compiler version and exit\n\n"
              << "Homepage: https://github.com/keryth/keryth\n";
}

} // namespace

int main(int argc, char* argv[]) {
    std::string_view program_name = (argc > 0 && argv[0] != nullptr) ? argv[0] : "keryth";

    if (argc == 1) {
        print_help(program_name);
        return 0;
    }

    std::string_view first_arg = argv[1];

    if (first_arg == "-v" || first_arg == "--version") {
        print_version();
        return 0;
    }

    if (first_arg == "-h" || first_arg == "--help") {
        print_help(program_name);
        return 0;
    }

    std::cerr << "keryth: error: unrecognized option or argument '" << first_arg << "'\n"
              << "Run '" << program_name << " --help' for usage information.\n";
    return 1;
}
