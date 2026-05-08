// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
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

#pragma once

#include <string>
#include <vector>

namespace lf {

// Per-mode inference parameters loaded from config file.
// Optional fields use sentinel values: -1 means "not set".
struct ModeConfig {
    // System prompt (inline string or from file)
    std::string system_prompt;       // inline system prompt text
    std::string system_prompt_file;  // path to file containing system prompt

    // Inference parameters (-1 / NaN means "not set, use llama.cpp default")
    float temp = -1.f;               // sampling temperature (--temp)
    float top_p = -1.f;             // top-p sampling (--top-p)
    float repeat_penalty = -1.f;    // repetition penalty (--repeat-penalty)
    int top_k = -1;                 // top-k sampling (--top-k)
    int n_predict = -1;             // max tokens to generate (-n)
    int ctx_size = -1;              // context window size (-c)
    int n_gpu_layers = -2;          // GPU layers to offload (-ngl); -2 = not set
};

// Full config loaded from ~/.config/llamafile/llamafile.yaml.
// Mode-specific sections override the [global] defaults.
struct LlamafileConfig {
    ModeConfig global;   // [global] section — fallback for all modes
    ModeConfig cli;      // [cli] section
    ModeConfig chat;     // [chat] section
    ModeConfig server;   // [server] section
    ModeConfig auto_;    // [auto] section (combined mode)
};

// Load config from ~/.config/llamafile/llamafile.yaml.
// Returns an empty/default config if the file does not exist.
// Prints a warning to stderr if the file exists but cannot be parsed.
LlamafileConfig load_config();

// Load config from an explicit file path.
// Prints an error to stderr and returns default config if the file cannot be opened.
LlamafileConfig load_config_from(const std::string &path);

// Print an annotated config template to stdout and exit(0).
void print_config_template();

// Resolve effective config for a given mode.
// Mode-specific values take precedence over global values.
ModeConfig resolve_mode_config(const LlamafileConfig &cfg,
                               const std::string &mode_name);

// Read the system prompt from a ModeConfig:
// - If system_prompt is non-empty, return it directly.
// - If system_prompt_file is set, read and return the file contents.
// - Otherwise return empty string.
// Prints a warning to stderr if the file cannot be read.
std::string resolve_system_prompt(const ModeConfig &mode);

// Inject config values as argv flags (only for params not already in argv).
// Appends to dst vector; caller must null-terminate when done.
void inject_config_flags(std::vector<char *> &dst,
                         const ModeConfig &mode,
                         int orig_argc,
                         char **orig_argv);

} // namespace lf
