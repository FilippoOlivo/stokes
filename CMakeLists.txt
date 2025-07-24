cmake_minimum_required(VERSION 3.13.4)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TARGET_NAME "stokes")
project(${TARGET_NAME} CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(deal.II 9.6.2 REQUIRED HINTS ${DEAL_II_DIR})


set(TARGET_SRC
    src/main.cpp
    src/stable_stokes.cpp
    src/stabilized_stokes.cpp
    src/navier_stokes.cpp
    src/boundary.cpp
)
add_compile_options(-fvisibility=hidden)
add_link_options(-fvisibility=hidden)
add_executable(${PROJECT_NAME} ${TARGET_SRC})
target_link_libraries(${PROJECT_NAME} ${DEAL_II_LIBRARIES})
deal_ii_initialize_cached_variables()
deal_ii_setup_target(${PROJECT_NAME})
