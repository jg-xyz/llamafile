#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# Turbo1Bit — TurboQuant KV cache compression library and inference tools
#
# Builds the TurboQuant core C library (portable, C11, links -lm) and the
# turbo1bit-infer tool that hooks into llama.cpp's KV cache for
# quantize-then-dequantize simulation of compressed inference.
#
# Metal GPU acceleration (Apple Silicon) is excluded from the cosmocc build
# because cosmocc targets cross-platform APE binaries; Metal requires the
# macOS SDK and ObjC runtime. Use `mise run turbo1bit:build` (cmake + clang)
# to build standalone tools with Metal acceleration.

PKGS += TURBO1BIT

# ==============================================================================
# TurboQuant core library (portable C11)
# ==============================================================================

TURBO1BIT_SRCS_C := \
	turbo1bit/src/turbo1bit_codebook.c \
	turbo1bit/src/turbo1bit_rotation.c \
	turbo1bit/src/turbo1bit_quantizer.c \
	turbo1bit/src/turbo1bit_kv_cache.c

TURBO1BIT_OBJS := \
	$(TURBO1BIT_SRCS_C:%.c=o/$(MODE)/%.c.o)

TURBO1BIT_HDRS := \
	turbo1bit/src/turbo1bit_codebook.h \
	turbo1bit/src/turbo1bit_rotation.h \
	turbo1bit/src/turbo1bit_quantizer.h \
	turbo1bit/src/turbo1bit_kv_cache.h

# Per-file compilation flags
$(TURBO1BIT_OBJS): \
		CFLAGS += \
			-iquotellamafile \
			-iquoteturbo1bit/src \
			-DTURBO1BIT_COSMOCC \
			-D_GNU_SOURCE

# Static library
o/$(MODE)/turbo1bit/turbo1bit.a: \
		$(TURBO1BIT_OBJS)
	$(AR) $(ARFLAGS) $@ $^

.PHONY: o/$(MODE)/turbo1bit
o/$(MODE)/turbo1bit: \
		o/$(MODE)/turbo1bit/turbo1bit.a \
		o/$(MODE)/turbo1bit/turbo1bit-infer

# ==============================================================================
# turbo1bit-infer tool
# ==============================================================================
# A standalone inference tool that applies TurboQuant compression to
# llama.cpp's KV cache after each decode step, enabling quality evaluation
# of compressed KV caches with any GGUF model.

TURBO1BIT_INFER_SRCS := \
	turbo1bit/tools/turbo1bit/turbo1bit_infer.cpp

$(TURBO1BIT_INFER_SRCS:%.cpp=o/$(MODE)/%.cpp.o): \
		CXXFLAGS += \
			-iquotellamafile \
			-iquoteturbo1bit/src \
			-iquotellama.cpp/common \
			-iquotellama.cpp/include \
			-iquotellama.cpp/src \
			-DTURBO1BIT_COSMOCC \
			-DTURBO1BIT_NO_METAL

o/$(MODE)/turbo1bit/turbo1bit-infer: \
		$(TURBO1BIT_INFER_SRCS:%.cpp=o/$(MODE)/%.cpp.o) \
		o/$(MODE)/turbo1bit/turbo1bit.a \
		o/$(MODE)/llama.cpp/libllama.a \
		o/$(MODE)/llamafile/libllama.a
	$(LINK.o) $^ $(LDLIBS) -lm -o $@

# ==============================================================================
# Dependency tracking
# ==============================================================================

$(TURBO1BIT_OBJS) \
$(TURBO1BIT_INFER_SRCS:%.cpp=o/$(MODE)/%.cpp.o): \
		$(COSMOCC)
