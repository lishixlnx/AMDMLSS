# cmake/project_functions.cmake
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Function to create Visual Studio folder structure matching file system
function(create_vs_folder_structure TARGET_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 ARG
        ""                              # Options
        ""                              # One value arguments
        "SOURCES;HEADERS"               # Multi-value arguments
    )
    
    # Get all source files for the target
    get_target_property(TARGET_SOURCES ${TARGET_NAME} SOURCES)
    
    if(TARGET_SOURCES)
        foreach(SOURCE_FILE ${TARGET_SOURCES})
            # Get the relative path from the source directory
            get_filename_component(SOURCE_PATH "${SOURCE_FILE}" DIRECTORY)
            file(RELATIVE_PATH REL_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${SOURCE_PATH}")
            
            # Skip if file is in current directory
            if(NOT REL_SOURCE_PATH OR REL_SOURCE_PATH STREQUAL ".")
                set(FOLDER_NAME "Source Files")
            else()
                # Create folder structure matching directory structure
                string(REPLACE "/" "\\" FOLDER_NAME "${REL_SOURCE_PATH}")
                
                # Determine if it's a header or source file
                get_filename_component(FILE_EXT "${SOURCE_FILE}" EXT)
                if(FILE_EXT MATCHES "\\.(h|hpp|hxx|inl)$")
                    set(FOLDER_NAME "Header Files\\${FOLDER_NAME}")
                else()
                    set(FOLDER_NAME "Source Files\\${FOLDER_NAME}")
                endif()
            endif()
            
            # Apply the source group
            source_group("${FOLDER_NAME}" FILES "${SOURCE_FILE}")
        endforeach()
    endif()
endfunction()

# Function to organize CMake files in Visual Studio
function(organize_cmake_files)
    # Find all CMake files in the project
    file(GLOB_RECURSE CMAKE_FILES
        "${CMAKE_SOURCE_DIR}/*.cmake"
        "${CMAKE_SOURCE_DIR}/CMakeLists.txt"
        "${CMAKE_SOURCE_DIR}/*/CMakeLists.txt"
    )
    
    foreach(CMAKE_FILE ${CMAKE_FILES})
        get_filename_component(CMAKE_PATH "${CMAKE_FILE}" DIRECTORY)
        file(RELATIVE_PATH REL_CMAKE_PATH "${CMAKE_SOURCE_DIR}" "${CMAKE_PATH}")
        
        if(NOT REL_CMAKE_PATH OR REL_CMAKE_PATH STREQUAL ".")
            set(FOLDER_NAME "CMake Files")
        else()
            string(REPLACE "/" "\\" FOLDER_NAME "CMake Files\\${REL_CMAKE_PATH}")
        endif()
        
        source_group("${FOLDER_NAME}" FILES "${CMAKE_FILE}")
    endforeach()
endfunction()

# Function to add standalone headers to Visual Studio
function(add_standalone_headers TARGET_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 ARG
        ""                              # Options
        "FOLDER"                        # One value arguments
        "HEADERS;PATHS"                 # Multi-value arguments
    )
    
    set(ALL_HEADERS "")
    
    # Add specific headers
    if(ARG_HEADERS)
        list(APPEND ALL_HEADERS ${ARG_HEADERS})
    endif()
    
    # Add headers from paths
    if(ARG_PATHS)
        foreach(HEADER_PATH ${ARG_PATHS})
            file(GLOB_RECURSE PATH_HEADERS 
                "${HEADER_PATH}/*.h"
                "${HEADER_PATH}/*.hpp"
                "${HEADER_PATH}/*.hxx"
                "${HEADER_PATH}/*.inl"
            )
            list(APPEND ALL_HEADERS ${PATH_HEADERS})
        endforeach()
    endif()
    
    # Remove duplicates
    if(ALL_HEADERS)
        list(REMOVE_DUPLICATES ALL_HEADERS)
        
        # Create custom target
        add_custom_target(${TARGET_NAME} SOURCES ${ALL_HEADERS})
        
        # Set folder
        if(ARG_FOLDER)
            set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "${ARG_FOLDER}")
        else()
            set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Headers")
        endif()
        
        # Apply source grouping
        foreach(HEADER_FILE ${ALL_HEADERS})
            get_filename_component(HEADER_PATH "${HEADER_FILE}" DIRECTORY)
            file(RELATIVE_PATH REL_HEADER_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${HEADER_PATH}")
            
            if(NOT REL_HEADER_PATH OR REL_HEADER_PATH STREQUAL ".")
                set(FOLDER_NAME "Header Files")
            else()
                string(REPLACE "/" "\\" FOLDER_NAME "Header Files\\${REL_HEADER_PATH}")
            endif()
            
            source_group("${FOLDER_NAME}" FILES "${HEADER_FILE}")
        endforeach()
    endif()
endfunction()

# Function to add a module with common settings
function(add_project_module NAME)
    cmake_parse_arguments(PARSE_ARGV 1 ARG
        "INSTALL_HEADERS"               # Options
        "TYPE;OUTPUT_NAME"              # One value arguments
        "SOURCES;HEADERS;DEPENDENCIES;INCLUDE_DIRS" # Multi-value arguments
    )

    # Validate required arguments
    if(NOT ARG_TYPE)
        message(FATAL_ERROR "Module type must be specified (STATIC/SHARED/INTERFACE)")
    endif()

    # Handle header-only libraries
    if(ARG_TYPE STREQUAL "INTERFACE")
        add_library(${NAME} INTERFACE)
        target_compile_features(${NAME} INTERFACE cxx_std_23)
        
        target_include_directories(${NAME}
            INTERFACE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
                ${ARG_INCLUDE_DIRS}
        )
        
        if(ARG_DEPENDENCIES)
            target_link_libraries(${NAME} INTERFACE ${ARG_DEPENDENCIES})
        endif()
        
        # Enhanced Visual Studio folder structure for interface libraries
        set(ALL_HEADERS "")
        
        # Collect headers from ARG_HEADERS
        if(ARG_HEADERS)
            file(GLOB_RECURSE INTERFACE_HEADERS ${ARG_HEADERS})
            list(APPEND ALL_HEADERS ${INTERFACE_HEADERS})
        endif()
        
        # Also collect headers from include directory automatically
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
            file(GLOB_RECURSE INCLUDE_HEADERS 
                "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hxx"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/*.inl"
            )
            list(APPEND ALL_HEADERS ${INCLUDE_HEADERS})
        endif()
        
        # Remove duplicates
        if(ALL_HEADERS)
            list(REMOVE_DUPLICATES ALL_HEADERS)
        endif()
        
        if(ALL_HEADERS)
            # Create a custom target to display headers in VS
            add_custom_target(${NAME}_headers SOURCES ${ALL_HEADERS})
            set_target_properties(${NAME}_headers PROPERTIES FOLDER "Libraries")
            
            # Apply detailed source grouping to the custom target
            foreach(HEADER_FILE ${ALL_HEADERS})
                get_filename_component(HEADER_PATH "${HEADER_FILE}" DIRECTORY)
                file(RELATIVE_PATH REL_HEADER_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${HEADER_PATH}")
                
                if(NOT REL_HEADER_PATH OR REL_HEADER_PATH STREQUAL ".")
                    set(FOLDER_NAME "Header Files")
                else()
                    # Create detailed folder structure
                    string(REPLACE "/" "\\" FOLDER_NAME "${REL_HEADER_PATH}")
                    set(FOLDER_NAME "Header Files\\${FOLDER_NAME}")
                endif()
                
                source_group("${FOLDER_NAME}" FILES "${HEADER_FILE}")
            endforeach()
        endif()
        
        return()
    endif()

    # Create target for non-interface libraries
    if(ARG_SOURCES)
        file(GLOB_RECURSE MODULE_SOURCES ${ARG_SOURCES})
    endif()
    
    if(ARG_HEADERS)
        file(GLOB_RECURSE MODULE_HEADERS ${ARG_HEADERS})
    endif()
    
    # Also automatically find headers in include directory
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
        file(GLOB_RECURSE INCLUDE_HEADERS 
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hxx"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.inl"
        )
        list(APPEND MODULE_HEADERS ${INCLUDE_HEADERS})
    endif()
    
    # Remove duplicates
    if(MODULE_HEADERS)
        list(REMOVE_DUPLICATES MODULE_HEADERS)
    endif()

    add_library(${NAME} ${ARG_TYPE}
        ${MODULE_SOURCES}
        ${MODULE_HEADERS}
    )

    # Set modern C++ standard
    target_compile_features(${NAME} PRIVATE cxx_std_23)

    # Add dependencies if any
    if(ARG_DEPENDENCIES)
        target_link_libraries(${NAME} PUBLIC ${ARG_DEPENDENCIES})
    endif()

    # Add include directories
    target_include_directories(${NAME}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
            ${ARG_INCLUDE_DIRS}
    )

    # Set output name if specified
    if(ARG_OUTPUT_NAME)
        set_target_properties(${NAME} PROPERTIES
            OUTPUT_NAME ${ARG_OUTPUT_NAME}
        )
    endif()

    # Create Visual Studio folder structure matching file system
    create_vs_folder_structure(${NAME})
    
    # Set the target folder in VS Solution Explorer
    set_target_properties(${NAME} PROPERTIES
        FOLDER "Libraries"
    )

    # Install headers if requested
    if(ARG_INSTALL_HEADERS)
        install(
            DIRECTORY include/
            DESTINATION include
            FILES_MATCHING 
                PATTERN "*.h"
                PATTERN "*.hpp"
                PATTERN "*.hxx"
        )
    endif()

    # Note: Target installation is handled manually in each module's CMakeLists.txt
endfunction()

# Function to add a test/sample program
function(add_project_test TEST_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 ARG
        ""                             # Options
        ""                             # One value arguments
        "DEPENDENCIES"                 # Multi-value arguments
    )
    
    # Get the source file (should be the first unparsed argument)
    list(GET ARG_UNPARSED_ARGUMENTS 0 SOURCE_FILE)
    
    # Create executable
    add_executable(${TEST_NAME} ${SOURCE_FILE})
    
    # Get file extension to determine language standard
    get_filename_component(FILE_EXT ${SOURCE_FILE} EXT)
    
    # Set appropriate standard based on file extension
    if(FILE_EXT STREQUAL ".cpp" OR FILE_EXT STREQUAL ".cxx" OR FILE_EXT STREQUAL ".cc")
        target_compile_features(${TEST_NAME} PRIVATE cxx_std_23)
    elseif(FILE_EXT STREQUAL ".c")
        target_compile_features(${TEST_NAME} PRIVATE c_std_17)
    endif()
    
    # Link dependencies
    if(ARG_DEPENDENCIES)
        target_link_libraries(${TEST_NAME} PRIVATE ${ARG_DEPENDENCIES})
        
        # Copy DLLs to sample output directory on Windows
        if(WIN32)
            foreach(DEPENDENCY ${ARG_DEPENDENCIES})
                add_custom_command(TARGET ${TEST_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:${DEPENDENCY}>
                    $<TARGET_FILE_DIR:${TEST_NAME}>
                    COMMENT "Copying ${DEPENDENCY} DLL to sample directory"
                )
            endforeach()
        endif()
    endif()
    
    # Set output directory for tests
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/samples
        FOLDER "Samples"
    )
    
    # Create Visual Studio folder structure for test files
    create_vs_folder_structure(${TEST_NAME})
endfunction()

# Function to add documentation files to Visual Studio
function(add_project_docs TARGET_NAME)
    cmake_parse_arguments(PARSE_ARGV 1 ARG
        ""                              # Options
        "FOLDER"                        # One value arguments
        "FILES;PATTERNS"                # Multi-value arguments
    )
    
    set(ALL_DOCS "")
    
    # Add specific files
    if(ARG_FILES)
        list(APPEND ALL_DOCS ${ARG_FILES})
    endif()
    
    # Add files matching patterns
    if(ARG_PATTERNS)
        foreach(PATTERN ${ARG_PATTERNS})
            file(GLOB_RECURSE PATTERN_FILES "${PATTERN}")
            list(APPEND ALL_DOCS ${PATTERN_FILES})
        endforeach()
    endif()
    
    # Remove duplicates
    if(ALL_DOCS)
        list(REMOVE_DUPLICATES ALL_DOCS)
        
        # Create custom target
        add_custom_target(${TARGET_NAME} SOURCES ${ALL_DOCS})
        
        # Set folder
        if(ARG_FOLDER)
            set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "${ARG_FOLDER}")
        else()
            set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Documentation")
        endif()
        
        # Apply source grouping
        foreach(DOC_FILE ${ALL_DOCS})
            get_filename_component(DOC_PATH "${DOC_FILE}" DIRECTORY)
            file(RELATIVE_PATH REL_DOC_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${DOC_PATH}")
            
            if(NOT REL_DOC_PATH OR REL_DOC_PATH STREQUAL ".")
                set(FOLDER_NAME "Documentation")
            else()
                string(REPLACE "/" "\\" FOLDER_NAME "Documentation\\${REL_DOC_PATH}")
            endif()
            
            source_group("${FOLDER_NAME}" FILES "${DOC_FILE}")
        endforeach()
    endif()
endfunction()

# Function to organize project files by type
function(organize_project_files)
    # Organize README files
    file(GLOB_RECURSE README_FILES
        "${CMAKE_SOURCE_DIR}/README*"
        "${CMAKE_SOURCE_DIR}/readme*"
        "${CMAKE_SOURCE_DIR}/*.md"
    )
    
    if(README_FILES)
        add_project_docs(Documentation_README
            FOLDER "Documentation"
            FILES ${README_FILES}
        )
    endif()
    
    # Organize license files
    file(GLOB_RECURSE LICENSE_FILES
        "${CMAKE_SOURCE_DIR}/LICENSE*"
        "${CMAKE_SOURCE_DIR}/license*"
        "${CMAKE_SOURCE_DIR}/COPYING*"
        "${CMAKE_SOURCE_DIR}/COPYRIGHT*"
    )
    
    if(LICENSE_FILES)
        add_project_docs(Documentation_License
            FOLDER "Documentation"
            FILES ${LICENSE_FILES}
        )
    endif()
    
    # Organize build scripts
    file(GLOB_RECURSE BUILD_SCRIPTS
        "${CMAKE_SOURCE_DIR}/*.sh"
        "${CMAKE_SOURCE_DIR}/*.bat"
        "${CMAKE_SOURCE_DIR}/*.ps1"
        "${CMAKE_SOURCE_DIR}/scripts/*"
    )
    
    if(BUILD_SCRIPTS)
        add_project_docs(Build_Scripts
            FOLDER "Scripts"
            FILES ${BUILD_SCRIPTS}
        )
    endif()
endfunction()

# Function to create a comprehensive project organization
function(setup_vs_project_structure)
    # Enable folder organization
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    
    # Organize CMake files
    organize_cmake_files()
    
    # Organize project files
    organize_project_files()
    
    message(STATUS "Visual Studio project structure organization enabled")
endfunction()

