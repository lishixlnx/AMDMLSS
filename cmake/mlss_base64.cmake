# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Base64 (RFC 4648) decoding implemented in pure CMake so no external tool
# (python/base64/certutil) is required at configure time on any supported
# platform. URLs are stored Base64-encoded in the build scripts rather than as
# plain text; this helper recovers the original value where it is consumed.
# Intended for short ASCII payloads such as dependency URLs.
include_guard(GLOBAL)

function(mlss_base64_decode OUT_VAR ENCODED)
    set(_alphabet "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
    string(REPLACE "=" "" _encoded "${ENCODED}")
    string(LENGTH "${_encoded}" _length)

    set(_decoded "")
    set(_buffer 0)
    set(_bits 0)

    if(_length GREATER 0)
        math(EXPR _last "${_length} - 1")
        foreach(_index RANGE ${_last})
            string(SUBSTRING "${_encoded}" ${_index} 1 _char)
            string(FIND "${_alphabet}" "${_char}" _value)
            if(_value LESS 0)
                message(FATAL_ERROR "mlss_base64_decode: invalid Base64 character '${_char}'.")
            endif()

            # Append the 6 bits of this symbol, then emit whole bytes as soon as
            # at least 8 bits have accumulated (most-significant bits first).
            math(EXPR _buffer "(${_buffer} << 6) | ${_value}")
            math(EXPR _bits "${_bits} + 6")
            if(_bits GREATER_EQUAL 8)
                math(EXPR _bits "${_bits} - 8")
                math(EXPR _byte "(${_buffer} >> ${_bits}) & 255")
                math(EXPR _buffer "${_buffer} & ((1 << ${_bits}) - 1)")
                string(ASCII ${_byte} _byte_char)
                string(APPEND _decoded "${_byte_char}")
            endif()
        endforeach()
    endif()

    set(${OUT_VAR} "${_decoded}" PARENT_SCOPE)
endfunction()
