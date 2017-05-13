cmake_minimum_required(VERSION 2.6.0)

project(libcudahilbertmaps)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIRS})

file(GLOB srcs *.cpp)
file(GLOB cuda Cuda/*.cu)

set(CUDA_ARCH_BIN "30 35 50 52 61" CACHE STRING "Specify GPU arch")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify virtual PTX arch")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
include(CudaComputeTargetFlags.cmake)                  
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")           
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false") 

CUDA_COMPILE(cuda_objs ${cuda})

find_package(PCL 1.3 REQUIRED COMPONENTS common io)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_CXX_FLAGS ${ADDITIONAL_CMAKE_CXX_FLAGS} "-O3 -msse2 -msse3 -Wall -std=c++11")

add_library(cudahilbertmaps SHARED
			${srcs}
			${cuda}
			${cuda_objs}
			${hdrs}
			${cuda_hdrs}
)

target_link_libraries(cudahilbertmaps
					  ${Eigen_LIBRARIES}
					  ${CUDA_LIBRARIES}
                      ${PCL_COMMON_LIBRARIES}
                      ${PCL_IO_LIBRARIES}
)

INSTALL(TARGETS cudahilbertmaps
			RUNTIME DESTINATION bin
			LIBRARY DESTINATION lib
			ARCHIVE DESTINATION lib
)

add_executable(output
			${srcs}
			${cuda}
			${cuda_objs}
			${hdrs}
			${cuda_hdrs}
)

target_link_libraries(output
					  ${Eigen_LIBRARIES}
					  ${CUDA_LIBRARIES}
                      ${PCL_COMMON_LIBRARIES}
                      ${PCL_IO_LIBRARIES}
)
