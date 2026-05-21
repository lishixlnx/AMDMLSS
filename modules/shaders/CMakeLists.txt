add_project_module(shaders
    TYPE STATIC
    SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"

    HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/shaders/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    DEPENDENCIES
        common
        core
    INCLUDE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/../.."
)

# Add C++20 module interface units (.cppm) for shader binaries
file(GLOB_RECURSE SHADER_CPPM_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cppm")
if(SHADER_CPPM_FILES)
    add_library(shader_modules STATIC)
    target_sources(shader_modules
        PUBLIC FILE_SET CXX_MODULES
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src"
        FILES ${SHADER_CPPM_FILES}
    )
    target_compile_features(shader_modules PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(shader_modules PRIVATE
            /MP1
            /bigobj
        )
        set_target_properties(shader_modules PROPERTIES
            MSVC_DEBUG_INFORMATION_FORMAT ""
        )
    endif()
    target_link_libraries(shaders PUBLIC shader_modules)
    set_target_properties(shaders PROPERTIES CXX_SCAN_FOR_MODULES ON)
endif()

# Install the target to export set
install(TARGETS shaders
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT amdmlss
)

if(TARGET shader_modules)
    install(TARGETS shader_modules
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT amdmlss
        FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_LIBDIR}/cxx_modules
    )
endif()
