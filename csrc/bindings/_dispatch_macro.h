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

// SG_DISPATCH: returns from the enclosing function. Use at the tail of
// a per-tensor wrapper (one launch per call).
#define SG_DISPATCH(METHOD, ...)                                              \
    do {                                                                      \
        const int sg_arch_ = ::sg::detect_arch();                             \
        switch (sg_arch_) {                                                   \
            case 80:  return ::sg::sm80::METHOD(__VA_ARGS__);                 \
            case 89:  return ::sg::sm89::METHOD(__VA_ARGS__);                 \
            case 90:  return ::sg::sm90::METHOD(__VA_ARGS__);                 \
            case 100: return ::sg::sm100::METHOD(__VA_ARGS__);                \
            case 103: return ::sg::sm103::METHOD(__VA_ARGS__);                \
            case 120: return ::sg::sm120::METHOD(__VA_ARGS__);                \
            case 942: return ::sg::gfx942::METHOD(__VA_ARGS__);               \
            case 950: return ::sg::gfx950::METHOD(__VA_ARGS__);               \
            default:                                                          \
                throw std::runtime_error(                                     \
                    std::string(#METHOD) + " dispatch: unsupported arch " +   \
                    std::to_string(sg_arch_));                                \
        }                                                                     \
    } while (0)

// SG_DISPATCH_CALL: same dispatch, no `return`. Use inside loops or
// wherever you need to dispatch and continue. The arch is re-detected
// each call (cached behind sg::detect_arch's lru-style caching).
#define SG_DISPATCH_CALL(METHOD, ...)                                         \
    do {                                                                      \
        const int sg_arch_ = ::sg::detect_arch();                             \
        switch (sg_arch_) {                                                   \
            case 80:  ::sg::sm80::METHOD(__VA_ARGS__);  break;                \
            case 89:  ::sg::sm89::METHOD(__VA_ARGS__);  break;                \
            case 90:  ::sg::sm90::METHOD(__VA_ARGS__);  break;                \
            case 100: ::sg::sm100::METHOD(__VA_ARGS__); break;                \
            case 103: ::sg::sm103::METHOD(__VA_ARGS__); break;                \
            case 120: ::sg::sm120::METHOD(__VA_ARGS__); break;                \
            case 942: ::sg::gfx942::METHOD(__VA_ARGS__); break;               \
            case 950: ::sg::gfx950::METHOD(__VA_ARGS__); break;               \
            default:                                                          \
                throw std::runtime_error(                                     \
                    std::string(#METHOD) + " dispatch: unsupported arch " +   \
                    std::to_string(sg_arch_));                                \
        }                                                                     \
    } while (0)
