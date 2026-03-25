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

# Install the target to export set
install(TARGETS shaders
    EXPORT ${PROJECT_NAME}-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
