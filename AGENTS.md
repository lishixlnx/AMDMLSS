<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# Project Guidelines

## Clean Code Guidelines

### Constants Over Magic Numbers
- Replace hard-coded values with named constants
- Use descriptive constant names that explain the value's purpose
- Keep constants at the top of the file or in a dedicated constants file
- Do not use equation and for integral type constants use decimal values.
- For floating points constants use floating point suffix

### Meaningful Names
- Variables, functions, and classes should reveal their purpose
- Names should explain why something exists and how it's used
- Avoid abbreviations unless they're universally understood

### Smart Comments
- Don't comment on what the code does - make the code self-documenting
- Use comments to explain why something is done a certain way
- Document APIs, complex algorithms, and non-obvious side effects

### Single Responsibility
- Each function should do exactly one thing
- Functions should be focused
- If a function needs a comment to explain what it does, it should be split

### DRY (Don't Repeat Yourself)
- Extract repeated code into reusable functions
- Share common logic through proper abstraction
- Maintain single sources of truth

### Clean Structure
- Keep related code together
- Organize code in a logical hierarchy
- Use consistent file and folder naming conventions

### Encapsulation
- Hide implementation details
- Expose clear interfaces
- Move nested conditionals into well-named functions

### Code Quality Maintenance
- Use the SOLID principle
- When needed fix technical debt early
- Leave code cleaner than you found it

### Unit Test
- Keep tests readable and maintainable
- Test edge cases and error conditions

### Version Control
- You are not allowed to use `git add`, `git commit` or `git push` unless explicitly asked.
- If explicitly asked:
 - write clear commit messages
 - make small, focused commits
 - use meaningful branch names



## C++ Guidelines

### Version
- This project use C++23. It may later use a newer version but shall not use an earlier version

### Files
- This project only use .hpp, .cpp, .hxx, and .cppm filenames. Other C++ extensions are not permitted.
- Files with the .cppm extension must be C++ modules.
- Classes, structures or functions declared for any modules in their include directory in a .hpp file (not .inl.hpp):
 - The definition shall be placed in a .cpp file for non template function, classes or structures, even for inline cases.
 - The definition shall be placed in a .inl.hpp for template function, classes or structures, even for inline cases.

### Namespace
- All classes and functions shall be at least under the `mlss` namespace
- Nested namespace should always be prefered
- In source files for local classes 
- For local classes, functions or structures that might be defined in a .inl.hpp file if the compiler is clang++ or g++ these structures should have a hidden visibility.

### Template and Compile Time deduction
- Concept must alway be prefered to SFINAE.
- If multiple functions have the same roles in multiple places for different types, a template shall be implemented.
- Use template and meta-template in every cases where it reduce the size of the implementation.
- Do not use template implementation to fix cases where a single type can be use.
- CRTP shall be use when it make sense.
- constexpr, consteval and constinit shall be use where it make sense.
- template specilization should be avoid if an "if constexpr (condition){...} else ..." can be used.

### Interface C/C++
- Under an Extern C scope all returned variables shall be assigned to an entry in the memory manager.

### Exceptions
- Exceptions shall be avoided as mush as possible.
- Exceptions shall be thrown only in context where an std::expected cannot be used.
- Exceptions shall be catch in the c_api module and transformed into a status flag.

### Views
- views, and container views, such as, but not limited to: std::string_view, shall be use wherever it make sense to use it.

## Project Build

### Platforms, compilers and configurations
- This project must build as a dynamic library on:
 - Windows, with:
  - clang++
  - visual studio 2022
  - visual studio 2026
 - Ubuntu, with:
  - clang++
  - g++
 - RHEL, with:
  - clang++
  - g++
  
## C Guidelines

### Version
- This project use C17. I may later use a newer version but shall not use an earlier version.

### Files
- This project only use .h file as header file.