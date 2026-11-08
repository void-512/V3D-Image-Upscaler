cmake_minimum_required(VERSION 3.20)
project(upscaler VERSION 1.0.0 LANGUAGES C CXX ASM)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Use an out-of-source build, preferably under ~/.cache/codex-tmp/upscaler-build")
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    message(FATAL_ERROR "This project targets Raspberry Pi 5 / ARM64 Linux with V3DV")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Vulkan REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(GLSLC_EXECUTABLE NAMES glslc HINTS "$ENV{VULKAN_SDK}/bin" REQUIRED)

foreach(dependency third_party/stb/stb_image.h third_party/stb/stb_image_write.h third_party/zlib-ng/CMakeLists.txt)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dependency}")
        message(FATAL_ERROR "Missing ${dependency}; run git submodule update --init --recursive")
    endif()
endforeach()

function(upscaler_zlib)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTING OFF)
    set(ZLIB_COMPAT ON)
    set(ZLIB_ALIASES OFF)
    set(SKIP_INSTALL_ALL ON)
    add_subdirectory(third_party/zlib-ng EXCLUDE_FROM_ALL)
    target_compile_options(zlib-ng PRIVATE -mcpu=cortex-a76)
endfunction()
upscaler_zlib()

set(generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${generated}")
set(manifest "${generated}/resources.txt")
set(resource_manifest "")
set(resource_dependencies "")

function(upscaler_model mode checkpoint reference)
    set(output "${generated}/models/${mode}.dat")
    set(converter "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_${mode}.py")
    set(checkpoint "${CMAKE_CURRENT_SOURCE_DIR}/models/checkpoints/${checkpoint}")
    add_custom_command(OUTPUT "${output}" "${generated}/models/${mode}.json"
        COMMAND "${Python3_EXECUTABLE}" -B "${converter}"
                --checkpoint "${checkpoint}" --output "${output}"
        DEPENDS "${checkpoint}" "${converter}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/${reference}"
        COMMENT "Converting ${mode} weights" VERBATIM)
    set(resource_manifest "${resource_manifest}${mode}/model|${output}\n" PARENT_SCOPE)
    list(APPEND resource_dependencies "${output}")
    set(resource_dependencies "${resource_dependencies}" PARENT_SCOPE)
endfunction()

upscaler_model(anime 4x_APISR_DAT_GAN_generator.pth dat.py)
upscaler_model(live RealESRGAN_x4plus.pth rrdbnet_arch.py)

function(upscaler_shader mode name source)
    set(output "${generated}/${mode}/${name}.spv")
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${mode}/${source}")
    set(dependencies "${input}")
    if(mode STREQUAL "anime")
        list(APPEND dependencies "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime/common.glsl")
    endif()
    add_custom_command(OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${generated}/${mode}"
        COMMAND "${GLSLC_EXECUTABLE}" -O --target-env=vulkan1.1 ${ARGN} "${input}" -o "${output}"
        DEPENDS ${dependencies}
        COMMENT "Compiling ${mode}/${name}" VERBATIM)
    set(resource_manifest "${resource_manifest}${mode}/${name}|${output}\n" PARENT_SCOPE)
    list(APPEND resource_dependencies "${output}")
    set(resource_dependencies "${resource_dependencies}" PARENT_SCOPE)
endfunction()

foreach(shader gemm_fast channel_fast channel_av softmax norm depthwise copy)
    upscaler_shader(anime "${shader}" "${shader}.comp")
endforeach()
upscaler_shader(anime window_qk window.comp -DQK)
upscaler_shader(anime window_av window.comp)
foreach(op 2 3 4 6 7 8 9 11)
    upscaler_shader(anime "op${op}" ops.comp "-DOP=${op}")
endforeach()
upscaler_shader(anime conv_win_in ../live/winograd.comp -DOP=0 -DANIME)
upscaler_shader(anime conv_win_mul ../live/winograd.comp -DOP=1 -DANIME)
upscaler_shader(anime conv_win_out ../live/winograd.comp -DOP=2 -DANIME)
upscaler_shader(live conv_fast conv.comp)
upscaler_shader(live conv_win_in winograd.comp -DOP=0)
upscaler_shader(live conv_win_mul winograd.comp -DOP=1)
upscaler_shader(live conv_win_out winograd.comp -DOP=2)
upscaler_shader(live pack pack.comp)
upscaler_shader(live output output.comp)
file(GENERATE OUTPUT "${manifest}" CONTENT "${resource_manifest}")

add_custom_command(OUTPUT "${generated}/resources.S" "${generated}/resources.cpp"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_resources.py"
            --manifest "${manifest}" --assembly "${generated}/resources.S" --index "${generated}/resources.cpp"
    DEPENDS "${manifest}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_resources.py" ${resource_dependencies}
    COMMENT "Embedding both models and all shaders" VERBATIM)

add_executable(upscaler
    src/main.cpp
    src/options.cpp src/upscale.cpp src/png_writer.cpp src/network.cpp src/gpu.cpp src/image_io.cpp
    src/networks/dat.cpp src/networks/rrdb.cpp
    "${generated}/resources.cpp" "${generated}/resources.S")
target_compile_features(upscaler PRIVATE cxx_std_20)
target_include_directories(upscaler PRIVATE src third_party/stb)
target_compile_definitions(upscaler PRIVATE UPSCALER_VERSION="${PROJECT_VERSION}")
target_compile_options(upscaler PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-mcpu=cortex-a76>")
target_link_libraries(upscaler PRIVATE Vulkan::Vulkan zlib-ng::zlibstatic)
install(TARGETS upscaler RUNTIME DESTINATION bin)
