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
        target_compile_options(shader_modules PRIVATE /MP1)
    endif()
    target_link_libraries(shaders PUBLIC shader_modules)
    set_target_properties(shaders PROPERTIES CXX_SCAN_FOR_MODULES ON)
endif()

# Install the target to export set
install(TARGETS shaders
    EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TARGET shader_modules)
    install(TARGETS shader_modules
        EXPORT ${PROJECT_NAME}-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_LIBDIR}/cxx_modules
    )
endif()
