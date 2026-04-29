// =====================================================================
//  _dispatch_macro.h — shared DISPATCH() macro for per-optimizer dispatchers
//
//  Each per-optimizer .cpp file uses DISPATCH(method, args...) to switch
//  on detect_arch() and call sg::<arch>::method(args...). The launcher
//  signature is whatever is forward-declared in that per-optimizer file.
//
//  Internal header; not part of the public API.
// =====================================================================

#pragma once

#include "bindings.h"

#include <stdexcept>
#include <string>

#define SG_DISPATCH(METHOD, ...)                                              \
    do {                                                                      \
        const int sg_arch_ = ::sg::detect_arch();                             \
        switch (sg_arch_) {                                                   \
            case 80:  return ::sg::sm80::METHOD(__VA_ARGS__);                 \
            case 90:  return ::sg::sm90::METHOD(__VA_ARGS__);                 \
            case 100: return ::sg::sm100::METHOD(__VA_ARGS__);                \
            case 942: return ::sg::gfx942::METHOD(__VA_ARGS__);               \
            default:                                                          \
                throw std::runtime_error(                                     \
                    std::string(#METHOD) + " dispatch: unsupported arch " +   \
                    std::to_string(sg_arch_));                                \
        }                                                                     \
    } while (0)
