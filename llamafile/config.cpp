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

//
// Llamafile config file loader.
//
// Reads ~/.config/llamafile/llamafile.yaml and populates a LlamafileConfig.
//
// Supported YAML subset:
//   - Section headers:  [global], [cli], [chat], [server], [auto]
//   - Key-value pairs:  key: value
//   - String values may optionally be quoted with single or double quotes
//   - '#' comments (rest of line ignored)
//   - Blank lines ignored
//
// Example config:
//
//   [global]
//   temp: 0.7
//   top_p: 0.9
//   system_prompt: You are a helpful assistant.
//
//   [cli]
//   n_predict: 512
//   system_prompt_file: ~/.config/llamafile/system_prompt.txt
//
//   [chat]
//   ctx_size: 8192
//

#include "config.h"

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace lf {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Strip inline '#' comment (not inside quotes).
static std::string strip_comment(const std::string &s) {
    bool in_single = false, in_double = false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '\'' && !in_double) { in_single = !in_single; continue; }
        if (c == '"'  && !in_single) { in_double = !in_double; continue; }
        if (c == '#' && !in_single && !in_double)
            return s.substr(0, i);
    }
    return s;
}

// Remove surrounding quotes from a value string.
static std::string unquote(const std::string &s) {
    if (s.size() >= 2) {
        char f = s.front(), b = s.back();
        if ((f == '"' && b == '"') || (f == '\'' && b == '\''))
            return s.substr(1, s.size() - 2);
    }
    return s;
}

// Expand leading ~/ to the home directory.
static std::string expand_home(const std::string &path) {
    if (path.size() >= 2 && path[0] == '~' && path[1] == '/') {
        const char *home = getenv("HOME");
        if (home && *home)
            return std::string(home) + path.substr(1);
    }
    return path;
}

// Return a pointer to the ModeConfig for a named section.
// Returns nullptr for unknown section names.
static ModeConfig *section_for(LlamafileConfig &cfg, const std::string &name) {
    if (name == "global") return &cfg.global;
    if (name == "cli")    return &cfg.cli;
    if (name == "chat")   return &cfg.chat;
    if (name == "server") return &cfg.server;
    if (name == "auto")   return &cfg.auto_;
    return nullptr;
}

// Apply a key=value pair to a ModeConfig.
static void apply_kv(ModeConfig &mc, const std::string &key, const std::string &value,
                     const std::string &filepath, int lineno) {
    if (key == "system_prompt") {
        mc.system_prompt = value;
    } else if (key == "system_prompt_file") {
        mc.system_prompt_file = expand_home(value);
    } else if (key == "temp" || key == "temperature") {
        mc.temp = (float)atof(value.c_str());
    } else if (key == "top_p") {
        mc.top_p = (float)atof(value.c_str());
    } else if (key == "top_k") {
        mc.top_k = atoi(value.c_str());
    } else if (key == "repeat_penalty") {
        mc.repeat_penalty = (float)atof(value.c_str());
    } else if (key == "n_predict" || key == "max_tokens") {
        mc.n_predict = atoi(value.c_str());
    } else if (key == "ctx_size" || key == "context_size") {
        mc.ctx_size = atoi(value.c_str());
    } else if (key == "n_gpu_layers" || key == "gpu_layers") {
        mc.n_gpu_layers = atoi(value.c_str());
    } else {
        fprintf(stderr, "llamafile: config: %s:%d: unknown key '%s' (ignored)\n",
                filepath.c_str(), lineno, key.c_str());
    }
}

// ---------------------------------------------------------------------------
// load_config / load_config_from
// ---------------------------------------------------------------------------

// Shared parsing logic: read an open stream into cfg.
static void parse_config_stream(std::istream &f, const std::string &filepath,
                                LlamafileConfig &cfg) {
    ModeConfig *current_section = &cfg.global;  // default to [global] before any header
    std::string line;
    int lineno = 0;

    while (std::getline(f, line)) {
        ++lineno;
        line = strip_comment(line);
        line = trim(line);
        if (line.empty()) continue;

        // Section header: [name]
        if (line.front() == '[' && line.back() == ']') {
            std::string name = trim(line.substr(1, line.size() - 2));
            current_section = section_for(cfg, name);
            if (!current_section) {
                fprintf(stderr, "llamafile: config: %s:%d: unknown section '[%s]' (ignored)\n",
                        filepath.c_str(), lineno, name.c_str());
            }
            continue;
        }

        // Key: value pair
        size_t colon = line.find(':');
        if (colon == std::string::npos) {
            fprintf(stderr, "llamafile: config: %s:%d: malformed line (ignored)\n",
                    filepath.c_str(), lineno);
            continue;
        }

        std::string key   = trim(line.substr(0, colon));
        std::string value = unquote(trim(line.substr(colon + 1)));

        if (key.empty()) continue;
        if (!current_section) continue;  // inside unknown section

        apply_kv(*current_section, key, value, filepath, lineno);
    }
}

LlamafileConfig load_config() {
    LlamafileConfig cfg;

    // Build path: ~/.config/llamafile/llamafile.yaml
    const char *config_home = getenv("XDG_CONFIG_HOME");
    std::string config_dir;
    if (config_home && *config_home) {
        config_dir = std::string(config_home) + "/llamafile";
    } else {
        const char *home = getenv("HOME");
        if (!home || !*home) return cfg;  // no home dir, skip
        config_dir = std::string(home) + "/.config/llamafile";
    }
    std::string filepath = config_dir + "/llamafile.yaml";

    std::ifstream f(filepath);
    if (!f.is_open()) {
        // File doesn't exist — silently return default config
        return cfg;
    }

    parse_config_stream(f, filepath, cfg);
    return cfg;
}

LlamafileConfig load_config_from(const std::string &path) {
    LlamafileConfig cfg;

    std::string expanded = expand_home(path);
    std::ifstream f(expanded);
    if (!f.is_open()) {
        fprintf(stderr, "llamafile: --config: cannot open '%s': %s\n",
                expanded.c_str(), strerror(errno));
        return cfg;
    }

    parse_config_stream(f, expanded, cfg);
    return cfg;
}

// ---------------------------------------------------------------------------
// print_config_template
// ---------------------------------------------------------------------------

void print_config_template() {
    printf(
        "# llamafile configuration file\n"
        "# Location: ~/.config/llamafile/llamafile.yaml\n"
        "#\n"
        "# Settings are organized into sections, one per execution mode.\n"
        "# [global] values apply to all modes unless overridden in a mode section.\n"
        "# CLI arguments always take precedence over config file values.\n"
        "#\n"
        "# Sections: [global]  [cli]  [chat]  [server]  [auto]\n"
        "#   global  - defaults for all modes\n"
        "#   cli     - llamafile --cli  (single prompt, then exit)\n"
        "#   chat    - llamafile --chat (interactive TUI)\n"
        "#   server  - llamafile --server (HTTP API only)\n"
        "#   auto    - llamafile (combined TUI + server, default mode)\n"
        "\n"
        "[global]\n"
        "\n"
        "# System prompt text (inline). Overridden by -p on the command line.\n"
        "# system_prompt: You are a helpful assistant.\n"
        "\n"
        "# Path to a file whose contents become the system prompt.\n"
        "# Ignored when system_prompt is set. Supports ~/.\n"
        "# system_prompt_file: ~/.config/llamafile/system_prompt.txt\n"
        "\n"
        "# Sampling temperature. 0 = deterministic, higher = more creative.\n"
        "# Equivalent to --temp. Default: 0\n"
        "# temp: 0.7\n"
        "\n"
        "# Top-p (nucleus) sampling threshold. Equivalent to --top-p.\n"
        "# top_p: 0.9\n"
        "\n"
        "# Top-k sampling. 0 = disabled. Equivalent to --top-k.\n"
        "# top_k: 40\n"
        "\n"
        "# Penalize repeated tokens. 1.0 = disabled. Equivalent to --repeat-penalty.\n"
        "# repeat_penalty: 1.1\n"
        "\n"
        "# Maximum number of tokens to generate. Equivalent to -n.\n"
        "# n_predict: 512\n"
        "\n"
        "# Context window size in tokens. Equivalent to -c.\n"
        "# ctx_size: 4096\n"
        "\n"
        "# Number of model layers to offload to GPU. -1 = all. Equivalent to -ngl.\n"
        "# n_gpu_layers: 99\n"
        "\n"
        "[cli]\n"
        "# Settings for --cli mode (single prompt -> response, then exit).\n"
        "# The system_prompt here becomes the system message in the chat template.\n"
        "# The user prompt is still provided via -p on the command line.\n"
        "\n"
        "# temp: 0.0\n"
        "# n_predict: 1024\n"
        "# system_prompt: You are a concise assistant. Answer briefly.\n"
        "\n"
        "[chat]\n"
        "# Settings for --chat mode (interactive TUI).\n"
        "\n"
        "# ctx_size: 8192\n"
        "# temp: 0.7\n"
        "# system_prompt: You are a helpful assistant.\n"
        "\n"
        "[server]\n"
        "# Settings for --server mode (OpenAI-compatible HTTP API).\n"
        "\n"
        "# ctx_size: 4096\n"
        "# n_gpu_layers: 99\n"
        "\n"
        "[auto]\n"
        "# Settings for the default combined mode (TUI chat + HTTP server).\n"
        "\n"
        "# ctx_size: 4096\n"
        "# system_prompt: You are a helpful assistant.\n"
    );
    exit(0);
}

// ---------------------------------------------------------------------------
// resolve_mode_config
// ---------------------------------------------------------------------------

ModeConfig resolve_mode_config(const LlamafileConfig &cfg,
                               const std::string &mode_name) {
    // Start with global defaults, then override with mode-specific values.
    ModeConfig result = cfg.global;

    const ModeConfig *mode = nullptr;
    if (mode_name == "cli")    mode = &cfg.cli;
    else if (mode_name == "chat")   mode = &cfg.chat;
    else if (mode_name == "server") mode = &cfg.server;
    else if (mode_name == "auto")   mode = &cfg.auto_;

    if (!mode) return result;

    // Override each field if the mode-specific value is set (non-sentinel).
    if (!mode->system_prompt.empty())
        result.system_prompt = mode->system_prompt;
    if (!mode->system_prompt_file.empty())
        result.system_prompt_file = mode->system_prompt_file;
    if (mode->temp >= 0.f)
        result.temp = mode->temp;
    if (mode->top_p >= 0.f)
        result.top_p = mode->top_p;
    if (mode->top_k >= 0)
        result.top_k = mode->top_k;
    if (mode->repeat_penalty >= 0.f)
        result.repeat_penalty = mode->repeat_penalty;
    if (mode->n_predict >= 0)
        result.n_predict = mode->n_predict;
    if (mode->ctx_size >= 0)
        result.ctx_size = mode->ctx_size;
    if (mode->n_gpu_layers >= -1)
        result.n_gpu_layers = mode->n_gpu_layers;

    return result;
}

// ---------------------------------------------------------------------------
// resolve_system_prompt
// ---------------------------------------------------------------------------

std::string resolve_system_prompt(const ModeConfig &mode) {
    // Inline prompt takes priority over file
    if (!mode.system_prompt.empty())
        return mode.system_prompt;

    if (!mode.system_prompt_file.empty()) {
        std::ifstream f(mode.system_prompt_file);
        if (!f.is_open()) {
            fprintf(stderr, "llamafile: config: cannot open system_prompt_file '%s': %s\n",
                    mode.system_prompt_file.c_str(), strerror(errno));
            return "";
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        std::string content = ss.str();
        // Strip trailing newline(s) — common in text files
        while (!content.empty() && (content.back() == '\n' || content.back() == '\r'))
            content.pop_back();
        return content;
    }

    return "";
}

// ---------------------------------------------------------------------------
// inject_config_flags
// ---------------------------------------------------------------------------

// Returns true if flag (and its value) already appear in orig_argv[1..orig_argc-1].
static bool argv_has_flag(int orig_argc, char **orig_argv, const char *flag) {
    for (int i = 1; i < orig_argc; ++i) {
        if (orig_argv[i] && strcmp(orig_argv[i], flag) == 0)
            return true;
    }
    return false;
}

// Storage for injected flag strings (must outlive the argv vector).
// Using static vector of strings so pointers remain valid.
static std::vector<std::string> g_injected_strings;

static void inject_flag_val(std::vector<char *> &dst,
                            int orig_argc, char **orig_argv,
                            const char *flag, const std::string &val) {
    if (argv_has_flag(orig_argc, orig_argv, flag)) return;
    g_injected_strings.push_back(std::string(flag));
    dst.push_back(const_cast<char *>(g_injected_strings.back().c_str()));
    g_injected_strings.push_back(val);
    dst.push_back(const_cast<char *>(g_injected_strings.back().c_str()));
}

void inject_config_flags(std::vector<char *> &dst,
                         const ModeConfig &mode,
                         int orig_argc,
                         char **orig_argv) {
    if (mode.temp >= 0.f) {
        inject_flag_val(dst, orig_argc, orig_argv, "--temp",
                        std::to_string(mode.temp));
    }
    if (mode.top_p >= 0.f) {
        inject_flag_val(dst, orig_argc, orig_argv, "--top-p",
                        std::to_string(mode.top_p));
    }
    if (mode.top_k >= 0) {
        inject_flag_val(dst, orig_argc, orig_argv, "--top-k",
                        std::to_string(mode.top_k));
    }
    if (mode.repeat_penalty >= 0.f) {
        inject_flag_val(dst, orig_argc, orig_argv, "--repeat-penalty",
                        std::to_string(mode.repeat_penalty));
    }
    if (mode.n_predict >= 0) {
        inject_flag_val(dst, orig_argc, orig_argv, "-n",
                        std::to_string(mode.n_predict));
    }
    if (mode.ctx_size >= 0) {
        inject_flag_val(dst, orig_argc, orig_argv, "-c",
                        std::to_string(mode.ctx_size));
    }
    if (mode.n_gpu_layers >= -1) {
        inject_flag_val(dst, orig_argc, orig_argv, "-ngl",
                        std::to_string(mode.n_gpu_layers));
    }
}

} // namespace lf
