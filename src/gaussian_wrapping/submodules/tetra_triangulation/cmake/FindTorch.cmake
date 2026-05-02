if(NOT DEFINED PYTHON_EXECUTABLE)
  execute_process(COMMAND python -c "import sys; print(sys.executable,end='')" OUTPUT_VARIABLE PYTHON_EXECUTABLE)
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" --version OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_PYTHON_VERSION)
  message(STATUS "Using ${TORCH_PYTHON_VERSION}")
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(f'{torch.__version__}',end='')" RESULT_VARIABLE STATUS OUTPUT_VARIABLE TORCH_VERSION)
  if(STATUS AND NOT STATUS EQUAL 0)
    message(FATAL_ERROR "Could not find torch library using python path: ${PYTHON_EXECUTABLE}")
  endif()
  message(STATUS "Found torch ${TORCH_VERSION}")
endif()
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import torch.utils.cpp_extension; print(f'{torch.utils.cpp_extension.CUDA_HOME}',end='')" RESULT_VARIABLE STATUS OUTPUT_VARIABLE CUDA_TOOLKIT_ROOT_DIR)
endif()
if(NOT DEFINED TORCH_LIBRARY_DIRS)
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import torch.utils.cpp_extension; print(';'.join(torch.utils.cpp_extension.library_paths(False)),end='')" COMMAND_ERROR_IS_FATAL ANY OUTPUT_VARIABLE TORCH_LIBRARY_DIRS)
endif()
if(NOT DEFINED TORCH_INCLUDE_DIRS)
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import torch.utils.cpp_extension; print(';'.join(torch.utils.cpp_extension.include_paths(False)),end='')" COMMAND_ERROR_IS_FATAL ANY OUTPUT_VARIABLE TORCH_INCLUDE_DIRS)
endif()
if(NOT DEFINED TORCH_CXX11_ABI)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(int(torch.compiled_with_cxx11_abi()), end='')"
    RESULT_VARIABLE STATUS
    OUTPUT_VARIABLE TORCH_CXX11_ABI)
  if(STATUS AND NOT STATUS EQUAL 0)
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI), end='')"
      COMMAND_ERROR_IS_FATAL ANY
      OUTPUT_VARIABLE TORCH_CXX11_ABI)
  endif()
endif()

if("$ENV{CUDAARCHS}" STREQUAL "")
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import torch.utils.cpp_extension;print(' '.join(sorted(set(x.split('_')[-1] for x in torch.utils.cpp_extension._get_cuda_arch_flags()))),end='')" COMMAND_ERROR_IS_FATAL ANY OUTPUT_VARIABLE CMAKE_CUDA_ARCHITECTURES)
endif()

# Setup everything
message(STATUS "Using torch libraries: ${TORCH_LIBRARY_DIRS}")
message(STATUS "Using torch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Using CUDA toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Using torch CXX11 ABI: ${TORCH_CXX11_ABI}")
include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_LIBRARY_DIRS})
set(TORCH_LIBRARIES c10 torch torch_cpu torch_python)
set(TORCH_COMPILE_OPTIONS -Wno-deprecated-declarations)
set(TORCH_COMPILE_DEFINITIONS _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
string(APPEND CMAKE_CUDA_FLAGS " -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -O3 --use_fast_math")
