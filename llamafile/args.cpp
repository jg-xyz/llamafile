// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
// Copyright 2026 Mozilla.ai
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "args.h"
#include "config.h"
#include "llamafile.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace lf {

// Static storage for filtered argv (persists after function returns)
static std::vector<char*> g_filtered_argv;

// Helper: returns true if arg is a llamafile-specific flag (not recognized by llama.cpp)
static bool is_llamafile_flag(const char* arg) {
    return strcmp(arg, "--server") == 0 ||
           strcmp(arg, "--chat") == 0 ||
           strcmp(arg, "--cli") == 0 ||
           strcmp(arg, "--gpu") == 0 ||
           strcmp(arg, "--ascii") == 0 ||
           strcmp(arg, "--nologo") == 0 ||
           strcmp(arg, "--nothink") == 0 ||
           strcmp(arg, "--turbo1bit") == 0 ||
           strcmp(arg, "--no-turbo1bit") == 0 ||
           strcmp(arg, "--version") == 0;
}

// Detect whether a GGUF model at the given path is a Bonsai / 1-bit model.
// Detection heuristic: Bonsai models embed "bonsai" or "prism" in the
// filename, OR are confirmed by GGUF architecture metadata containing
// "bitnet" or "1bit". For path-only detection (before model load) we
// check the filename; full detection happens in chatbot_main after model load.
static bool looks_like_1bit_model(const std::string& model_path) {
    if (model_path.empty()) return false;

    // Lowercase the basename for case-insensitive matching
    auto to_lower = [](std::string s) {
        for (char& c : s) c = (char)('A' <= c && c <= 'Z' ? c + 32 : c);
        return s;
    };

    // Extract basename (everything after the last / or \)
    size_t slash = model_path.find_last_of("/\\");
    std::string basename = (slash == std::string::npos)
        ? model_path
        : model_path.substr(slash + 1);
    std::string lower = to_lower(basename);

    // Common naming patterns for 1-bit models
    return lower.find("bonsai")  != std::string::npos ||
           lower.find("bitnet")  != std::string::npos ||
           lower.find("1bit")    != std::string::npos ||
           lower.find("1.58bit") != std::string::npos ||
           lower.find("prism")   != std::string::npos;
}

// Check if an argument already exists in argv (prevents double-injection)
static bool argv_has(const std::vector<char*>& argv_vec, const char* flag) {
    for (const char* a : argv_vec) {
        if (a && strcmp(a, flag) == 0) return true;
    }
    return false;
}

// Inject a Turbo1Bit flag+value pair into the filtered argv.
// flag and value must be string literals (static lifetime); their char* is
// stored directly without copying.
static void inject_flag(std::vector<char*>& dst, const char* flag, const char* value = nullptr) {
    if (!argv_has(dst, flag)) {
        dst.push_back(const_cast<char*>(flag));
        if (value) dst.push_back(const_cast<char*>(value));
    }
}

LlamafileArgs parse_llamafile_args(int argc, char** argv) {
    LlamafileArgs args;

    // Early GPU init must happen before we filter args
    // This reads --gpu and -ngl flags to set FLAG_gpu
    llamafile_early_gpu_init(argv);

    // Capture -p/--prompt value before filtering (needed for combined mode
    // where SERVER parsing excludes -p)
    // Note: Loop does not break early; if multiple -p flags are given,
    // the last occurrence wins (intentional for override flexibility)
    for (int i = 0; i < argc; ++i) {
        if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            args.system_prompt = argv[i + 1];
        }
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            args.model_path = argv[i + 1];
        }
    }

    // Determine execution mode from flags
    // Priority: explicit flags override defaults
    if (llamafile_has(argv, "--server")) {
        args.mode = ProgramMode::SERVER;
    } else if (llamafile_has(argv, "--chat")) {
        args.mode = ProgramMode::CHAT;
    } else if (llamafile_has(argv, "--cli")) {
        args.mode = ProgramMode::CLI;
    } else {
        // AUTO mode: will run combined chat + server
        args.mode = ProgramMode::AUTO;
    }

    // Check verbose flag
    FLAG_verbose = llamafile_has(argv, "--verbose") ? 1 : 0;

    // Check --nothink flag (filters thinking/reasoning content in CLI mode)
    FLAG_nothink = llamafile_has(argv, "--nothink");

    // Check logo flags
    FLAG_nologo = llamafile_has(argv, "--nologo");
    FLAG_ascii = llamafile_has(argv, "--ascii");

    // Turbo1Bit KV cache compression flag handling:
    //
    //   --turbo1bit      Explicitly enable Turbo1Bit (auto-injects --fa --ctk q4_0 --ctv q4_0)
    //   --no-turbo1bit   Explicitly disable Turbo1Bit (no injection even for 1-bit models)
    //
    // If neither flag is given and the model looks like a 1-bit model (Bonsai/BitNet),
    // Turbo1Bit is auto-enabled as the optimally efficient default.
    const bool explicit_enable  = llamafile_has(argv, "--turbo1bit");
    const bool explicit_disable = llamafile_has(argv, "--no-turbo1bit");

    if (explicit_disable) {
        FLAG_turbo1bit = false;
    } else if (explicit_enable) {
        FLAG_turbo1bit = true;
    } else {
        // Auto-detect: enable for Bonsai/BitNet models
        FLAG_turbo1bit = looks_like_1bit_model(args.model_path);
        if (FLAG_turbo1bit && FLAG_verbose) {
            fprintf(stderr, "llamafile: 1-bit model detected, auto-enabling Turbo1Bit "
                            "KV cache compression (--fa --ctk q4_0 --ctv q4_0)\n"
                            "llamafile: pass --no-turbo1bit to disable\n");
        }
    }

    // Load config file and resolve mode-specific settings.
    // CLI args take precedence: config values are only injected if the
    // corresponding flag is not already present in argv.
    {
        // Check for --config [PATH]: if PATH is omitted, print template and exit.
        // If PATH is provided, load from that file instead of the default location.
        LlamafileConfig cfg;
        bool found_config_flag = false;
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--config") != 0) continue;
            found_config_flag = true;
            // Next arg is the path only if it doesn't start with '-' and exists
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg = load_config_from(argv[i + 1]);
            } else {
                print_config_template();  // exits
            }
            break;
        }
        if (!found_config_flag)
            cfg = load_config();
        std::string mode_name;
        switch (args.mode) {
            case ProgramMode::CLI:    mode_name = "cli";    break;
            case ProgramMode::CHAT:   mode_name = "chat";   break;
            case ProgramMode::SERVER: mode_name = "server"; break;
            case ProgramMode::AUTO:   mode_name = "auto";   break;
        }
        args.mode_config = resolve_mode_config(cfg, mode_name);

        // If no -p/--prompt was given on the CLI, use the config system prompt.
        bool cli_has_prompt = false;
        for (int i = 0; i < argc; ++i) {
            if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
                cli_has_prompt = true;
                break;
            }
        }
        if (!cli_has_prompt) {
            std::string cfg_prompt = resolve_system_prompt(args.mode_config);
            if (!cfg_prompt.empty())
                args.system_prompt = cfg_prompt;
        }
    }

    // Filter out llamafile-specific arguments
    // These are not recognized by llama.cpp and would cause errors
    g_filtered_argv.clear();

    for (int i = 0; i < argc; ++i) {
        const char* arg = argv[i];

        // Skip llamafile-specific flags
        if (is_llamafile_flag(arg)) {
            // --gpu takes a value argument, skip it too
            if (strcmp(arg, "--gpu") == 0 && i + 1 < argc) {
                ++i;
            }
            continue;
        }

        // --config [PATH]: skip flag and optional path argument
        if (strcmp(arg, "--config") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-')
                ++i;
            continue;
        }

        // Keep this argument
        g_filtered_argv.push_back(argv[i]);
    }

    // Inject config-derived inference parameter flags (only for unset params).
    inject_config_flags(g_filtered_argv, args.mode_config, argc, argv);

    // For chat/auto/server modes: inject -p <system_prompt> from config when
    // the caller didn't supply -p on the CLI. CLI mode handles system_prompt
    // separately via cli_main()'s explicit parameter.
    if (args.mode != ProgramMode::CLI && !args.system_prompt.empty()) {
        bool has_p = false;
        for (int i = 0; i < argc; ++i) {
            if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
                has_p = true;
                break;
            }
        }
        if (!has_p) {
            static std::vector<std::string> g_prompt_storage;
            g_prompt_storage.push_back(args.system_prompt);
            g_filtered_argv.push_back(const_cast<char*>("-p"));
            g_filtered_argv.push_back(const_cast<char*>(g_prompt_storage.back().c_str()));
        }
    }

    // Inject Turbo1Bit flags into llama.cpp argv when enabled.
    // --fa enables Flash Attention (required for quantized KV cache).
    // --ctk / --ctv set the KV cache tensor types to Q4_0 for ~2.65x
    // memory reduction at 5% perplexity cost (safe for 1-bit models).
    if (FLAG_turbo1bit) {
        inject_flag(g_filtered_argv, "-fa", "on");
        inject_flag(g_filtered_argv, "-ctk", "q4_0");
        inject_flag(g_filtered_argv, "-ctv", "q4_0");
    }

    // Null-terminate argv array (required by convention)
    g_filtered_argv.push_back(nullptr);

    args.llama_argc = static_cast<int>(g_filtered_argv.size()) - 1;
    args.llama_argv = g_filtered_argv.data();

    return args;
}

} // namespace lf

