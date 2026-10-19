#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "keryth/ir_builder.hpp"
#include "keryth/lexer.hpp"
#include "keryth/optimizer.hpp"
#include "keryth/parser.hpp"
#include "keryth/runtime.hpp"
#include "keryth/type_checker.hpp"
#include "keryth/version.hpp"

namespace {

void print_version() {
    std::cout << "Keryth Compiler v" << keryth::get_version() << "\n"
              << keryth::get_tagline() << "\n";
}

void print_help(std::string_view program_name) {
    std::cout << "Usage: " << program_name << " [options] [source_file.kr]\n\n"
              << "Keryth: An AI-native programming language and optimizing compiler.\n\n"
              << "Commands & Options:\n"
              << "  -h, --help               Show this help message and exit\n"
              << "  -v, --version            Show compiler version and exit\n"
              << "  --dump-ast <file.kr>     Print abstract syntax tree\n"
              << "  --dump-ir <file.kr>      Print unoptimized SSA intermediate representation\n"
              << "  --dump-opt-ir <file.kr>  Print optimized SSA intermediate representation\n"
              << "  run <file.kr> [--opt]    Compile and execute program top-level function\n\n"
              << "Homepage: https://github.com/keryth/keryth\n";
}

std::string read_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "keryth: error: cannot open file '" << filepath << "'\n";
        return "";
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

} // namespace

int main(int argc, char* argv[]) {
    std::string_view program_name = (argc > 0 && argv[0] != nullptr) ? argv[0] : "keryth";

    if (argc == 1) {
        print_help(program_name);
        return 0;
    }

    std::string_view mode = argv[1];

    if (mode == "-v" || mode == "--version") {
        print_version();
        return 0;
    }

    if (mode == "-h" || mode == "--help") {
        print_help(program_name);
        return 0;
    }

    if (mode == "--dump-ast" && argc >= 3) {
        std::string src = read_file(argv[2]);
        if (src.empty()) return 1;

        keryth::Lexer lexer(src, argv[2]);
        keryth::Parser parser(lexer.tokenize_all());
        auto program = parser.parse_program();

        if (parser.has_errors()) {
            for (const auto& diag : parser.diagnostics()) {
                std::cerr << "error: " << diag.message << " at " << diag.span.to_string() << "\n";
            }
            return 1;
        }

        std::cout << "Parsed Program with " << program->functions.size() << " function(s):\n";
        for (const auto& fn : program->functions) {
            std::cout << "  fn " << fn->name << " (params: " << fn->params.size()
                      << ", stmts: " << fn->body.size() << ")\n";
        }
        return 0;
    }

    if (mode == "--dump-ir" && argc >= 3) {
        std::string src = read_file(argv[2]);
        if (src.empty()) return 1;

        keryth::Lexer lexer(src, argv[2]);
        keryth::Parser parser(lexer.tokenize_all());
        auto program = parser.parse_program();

        if (parser.has_errors()) {
            for (const auto& diag : parser.diagnostics()) {
                std::cerr << "error: " << diag.message << " at " << diag.span.to_string() << "\n";
            }
            return 1;
        }

        keryth::TypeChecker checker;
        if (!checker.check_program(*program)) {
            for (const auto& diag : checker.diagnostics()) {
                std::cerr << "type error: " << diag.message << " at " << diag.span.to_string() << "\n";
            }
            return 1;
        }

        keryth::ASTLowerer lowerer(checker);
        keryth::IRModule module = lowerer.lower_program(*program);
        std::cout << keryth::dump_ir(module);
        return 0;
    }

    if (mode == "--dump-opt-ir" && argc >= 3) {
        std::string src = read_file(argv[2]);
        if (src.empty()) return 1;

        keryth::Lexer lexer(src, argv[2]);
        keryth::Parser parser(lexer.tokenize_all());
        auto program = parser.parse_program();

        if (parser.has_errors()) {
            for (const auto& diag : parser.diagnostics()) {
                std::cerr << "error: " << diag.message << " at " << diag.span.to_string() << "\n";
            }
            return 1;
        }

        keryth::TypeChecker checker;
        if (!checker.check_program(*program)) {
            for (const auto& diag : checker.diagnostics()) {
                std::cerr << "type error: " << diag.message << " at " << diag.span.to_string() << "\n";
            }
            return 1;
        }

        keryth::ASTLowerer lowerer(checker);
        keryth::IRModule module = lowerer.lower_program(*program);

        keryth::PassManager pm;
        pm.add_default_passes();
        pm.run(module);

        std::cout << keryth::dump_ir(module);
        return 0;
    }

    if (mode == "run" && argc >= 3) {
        std::string src = read_file(argv[2]);
        if (src.empty()) return 1;

        bool optimize = (argc >= 4 && std::string_view(argv[3]) == "--opt");

        keryth::Lexer lexer(src, argv[2]);
        keryth::Parser parser(lexer.tokenize_all());
        auto program = parser.parse_program();

        if (parser.has_errors()) {
            for (const auto& diag : parser.diagnostics()) {
                std::cerr << "error: " << diag.message << " at " << diag.span.to_string() << "\n";
            }
            return 1;
        }

        keryth::TypeChecker checker;
        if (!checker.check_program(*program)) {
            for (const auto& diag : checker.diagnostics()) {
                std::cerr << "type error: " << diag.message << " at " << diag.span.to_string() << "\n";
            }
            return 1;
        }

        keryth::ASTLowerer lowerer(checker);
        keryth::IRModule module = lowerer.lower_program(*program);

        if (optimize) {
            keryth::PassManager pm;
            pm.add_default_passes();
            pm.run(module);
        }

        if (module.functions.empty()) {
            std::cerr << "keryth: error: no functions found to execute in '" << argv[2] << "'\n";
            return 1;
        }

        const auto& fn = module.functions[0];
        std::cout << "Executing function @" << fn.name << " (" << (optimize ? "optimized" : "unoptimized") << ")...\n";

        // Create zero-initialized inputs based on parameter types
        std::vector<keryth::RuntimeValue> args;
        for (keryth::ValueId pid : fn.parameters) {
            const auto* val = fn.get_value(pid);
            if (val && val->type) {
                if (val->type->is_tensor()) {
                    args.push_back(keryth::Tensor(val->type->dimensions(), 1.0f));
                } else if (val->type->element_kind() == keryth::ScalarKind::I32) {
                    args.push_back(int32_t{1});
                } else if (val->type->element_kind() == keryth::ScalarKind::F32) {
                    args.push_back(1.0f);
                } else {
                    args.push_back(true);
                }
            }
        }

        keryth::Interpreter interp;
        auto res = interp.execute_function(fn, args);

        if (std::holds_alternative<keryth::Tensor>(res)) {
            const auto& t = std::get<keryth::Tensor>(res);
            std::cout << "Result: Tensor with shape [";
            for (size_t i = 0; i < t.shape().size(); ++i) {
                std::cout << t.shape()[i];
                if (i + 1 < t.shape().size()) std::cout << ", ";
            }
            std::cout << "] (numel=" << t.numel() << ")\n";
        } else if (std::holds_alternative<int32_t>(res)) {
            std::cout << "Result: " << std::get<int32_t>(res) << "\n";
        } else if (std::holds_alternative<float>(res)) {
            std::cout << "Result: " << std::get<float>(res) << "\n";
        }

        return 0;
    }

    std::cerr << "keryth: error: unrecognized command or arguments '" << mode << "'\n"
              << "Run '" << program_name << " --help' for usage information.\n";
    return 1;
}
