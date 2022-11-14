find_package(OpenCV REQUIRED core imgcodecs)

add_executable(
	cuda_histogram
	cuda_histogram.cu)

set_target_properties(cuda_histogram PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(cuda_histogram PUBLIC cxx_std_17)
target_include_directories(cuda_histogram PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cuda_histogram PRIVATE ${OpenCV_LIBS})

install(TARGETS cuda_histogram DESTINATION ${CMAKE_INSTALL_PREFIX})

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
