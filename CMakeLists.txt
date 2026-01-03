cmake_minimum_required(VERSION 3.18)
project(CudaVisionHack LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui imgproc)

add_executable(cuda_vision main.cpp kernel.cu)

target_link_libraries(cuda_vision PRIVATE ${OpenCV_LIBS})

target_compile_options(cuda_vision PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
