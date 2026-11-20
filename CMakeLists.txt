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

find_path(UPSCALER_STB_INCLUDE NAMES stb_image.h
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb" PATH_SUFFIXES stb REQUIRED)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tomlplusplus/include/toml++/toml.hpp")
    message(FATAL_ERROR "toml++ is missing. Run git submodule update --init third_party/tomlplusplus.")
endif()
set(UPSCALER_MESA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/mesa" CACHE PATH "Mesa 26.2.7 source checkout")
set(UPSCALER_MESA_JOBS 3 CACHE STRING "Maximum parallel jobs inside the Mesa build")
set(UPSCALER_MODEL_BUNDLE "" CACHE FILEPATH "Optional legacy upscaler executable containing v2 model assets")
set(UPSCALER_ANIME_MODEL "" CACHE FILEPATH "Optional preconverted anime v2 model")
set(UPSCALER_LIVE_MODEL "" CACHE FILEPATH "Optional preconverted live v2 model")
option(UPSCALER_DOWNLOAD_MODELS "Fetch pinned official checkpoints when missing" ON)
set(UPSCALER_RUNTIME_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "Staging root for bin, lib and share")

find_program(MESON_EXECUTABLE meson REQUIRED)
find_program(NINJA_EXECUTABLE ninja REQUIRED)
set(icd_library "${UPSCALER_RUNTIME_DIR}/lib/libvulkan_broadcom.so")
set(icd_manifest "${UPSCALER_RUNTIME_DIR}/share/vulkan/icd.d/broadcom_custom.json")
set(mesa_patch "${CMAKE_CURRENT_SOURCE_DIR}/patches/mesa/0001-v3dv-upscaler-packed-fp16.patch")
set(mesa_builder "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_v3dv.py")
add_custom_command(OUTPUT "${icd_library}" "${icd_manifest}"
    COMMAND "${Python3_EXECUTABLE}" -B "${mesa_builder}"
        --source "${UPSCALER_MESA_SOURCE}" --build "${CMAKE_CURRENT_BINARY_DIR}/private-mesa"
        --patch "${mesa_patch}" --library "${icd_library}" --manifest "${icd_manifest}"
        --meson "${MESON_EXECUTABLE}" --ninja "${NINJA_EXECUTABLE}" --jobs "${UPSCALER_MESA_JOBS}"
    COMMAND "${CMAKE_STRIP}" --strip-unneeded "${icd_library}"
    DEPENDS "${mesa_builder}" "${mesa_patch}" "${UPSCALER_MESA_SOURCE}/VERSION"
    COMMENT "Building private Mesa 26.2.7 V3DV ICD" VERBATIM)
add_custom_target(private_v3dv DEPENDS "${icd_library}" "${icd_manifest}")
install(FILES "${icd_library}" DESTINATION lib)
install(FILES "${icd_manifest}" DESTINATION share/vulkan/icd.d)

if(NOT EXISTS "${UPSCALER_RUNTIME_DIR}/config.toml")
    configure_file(config.toml "${UPSCALER_RUNTIME_DIR}/config.toml" COPYONLY)
endif()
install(FILES config.toml DESTINATION .)

function(upscaler_zlib)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTING OFF)
    set(ZLIB_COMPAT ON)
    set(ZLIB_ALIASES OFF)
    set(SKIP_INSTALL_ALL ON)
    add_subdirectory(third_party/zlib-ng EXCLUDE_FROM_ALL)
    target_compile_options(zlib-ng PRIVATE -mcpu=cortex-a76)
endfunction()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/zlib-ng/CMakeLists.txt")
    upscaler_zlib()
    set(zlib_target zlib-ng::zlibstatic)
else()
    find_package(ZLIB REQUIRED)
    set(zlib_target ZLIB::ZLIB)
endif()

set(generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${generated}")
set(manifest "${generated}/resources.txt")
set(resource_manifest "")
set(resource_dependencies "")

function(upscaler_model mode checkpoint reference local_model url digest)
    set(directory "${generated}/models")
    set(output "${directory}/${mode}.dat")
    set(packed "${UPSCALER_RUNTIME_DIR}/share/upscaler/models/${mode}.packed.dat")
    set(converter "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_${mode}.py")
    set(packager "${CMAKE_CURRENT_SOURCE_DIR}/tools/package_model.py")
    if(local_model)
        add_custom_command(OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${directory}"
            COMMAND "${CMAKE_COMMAND}" -E copy "${local_model}" "${output}"
            DEPENDS "${local_model}" VERBATIM)
    elseif(UPSCALER_MODEL_BUNDLE)
        add_custom_command(OUTPUT "${output}"
            COMMAND "${Python3_EXECUTABLE}" -B "${packager}" "${mode}" "${UPSCALER_MODEL_BUNDLE}" "${output}" --extract
            DEPENDS "${packager}" "${UPSCALER_MODEL_BUNDLE}" VERBATIM)
    else()
        set(file "${CMAKE_CURRENT_SOURCE_DIR}/models/checkpoints/${checkpoint}")
        if(NOT EXISTS "${file}")
            if(NOT UPSCALER_DOWNLOAD_MODELS)
                message(FATAL_ERROR "Missing ${file}. Supply the checkpoint, a v2 model, or UPSCALER_MODEL_BUNDLE.")
            endif()
            set(file "${generated}/checkpoints/${checkpoint}")
            add_custom_command(OUTPUT "${file}"
                COMMAND "${CMAKE_COMMAND}" -DURL=${url} -DSHA256=${digest} -DOUTPUT=${file}
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchModel.cmake"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchModel.cmake"
                COMMENT "Downloading ${mode} checkpoint" VERBATIM)
        endif()
        add_custom_command(OUTPUT "${output}"
            BYPRODUCTS "${directory}/${mode}.json"
            COMMAND "${Python3_EXECUTABLE}" -B "${converter}" --checkpoint "${file}" --output "${output}"
            DEPENDS "${file}" "${converter}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/${reference}"
            COMMENT "Converting ${mode} weights" VERBATIM)
    endif()
    add_custom_command(OUTPUT "${packed}"
        COMMAND "${Python3_EXECUTABLE}" -B "${packager}" "${mode}" "${output}" "${packed}"
        DEPENDS "${packager}" "${output}" COMMENT "Prepacking ${mode} weights" VERBATIM)
    install(FILES "${packed}" DESTINATION share/upscaler/models)
    set(model_outputs "${model_outputs};${packed}" PARENT_SCOPE)
endfunction()

upscaler_model(anime 4x_APISR_DAT_GAN_generator.pth dat.py "${UPSCALER_ANIME_MODEL}"
    https://github.com/Kiteretsu77/APISR/releases/download/v0.3.0/4x_APISR_DAT_GAN_generator.pth
    cc625c4ec5242a57e46f0941a3c36e3a731dcd859750c5f72a5251045b1e6d72)
upscaler_model(live RealESRGAN_x4plus.pth rrdbnet_arch.py "${UPSCALER_LIVE_MODEL}"
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
    4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1)
add_custom_target(upscaler_models DEPENDS ${model_outputs})

function(upscaler_shader mode name source)
    set(output "${generated}/${mode}/${name}.spv")
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${mode}/${source}")
    set(dependencies "${input}")
    if(mode STREQUAL "anime")
        list(APPEND dependencies "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime/common.glsl")
    endif()
    add_custom_command(OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${generated}/${mode}"
        COMMAND "${GLSLC_EXECUTABLE}" -O --target-env=vulkan1.1 -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders/common"
            -MD -MF "${output}.d" ${ARGN} "${input}" -o "${output}"
        DEPENDS ${dependencies}
        DEPFILE "${output}.d"
        COMMENT "Compiling ${mode}/${name}" VERBATIM)
    set(resource_manifest "${resource_manifest}${mode}/${name}|${output}\n" PARENT_SCOPE)
    list(APPEND resource_dependencies "${output}")
    set(resource_dependencies "${resource_dependencies}" PARENT_SCOPE)
endfunction()

foreach(shader gemm_fast channel_fast channel_av norm depthwise copy)
    upscaler_shader(anime "${shader}" "${shader}.comp")
endforeach()
upscaler_shader(anime window_qk window.comp -DQK)
foreach(op 2 3 6 7 8 9)
    upscaler_shader(anime "op${op}" ops.comp "-DOP=${op}")
endforeach()
upscaler_shader(anime conv_win_in ../live/winograd.comp -DOP=0 -DANIME)
upscaler_shader(anime conv_win_mul ../live/winograd.comp -DOP=1 -DANIME)
upscaler_shader(anime conv_win_out ../live/winograd.comp -DOP=2 -DANIME)
upscaler_shader(anime gemm_fast_half gemm_fast.comp -DPACKED_DAT)
upscaler_shader(anime norm_half norm.comp -DNARROW)
upscaler_shader(anime depthwise_half depthwise.comp -DNARROW)
upscaler_shader(anime op4_half ops.comp -DOP=4 -DNARROW)
upscaler_shader(anime softmax_half softmax.comp -DNARROW)
upscaler_shader(anime window_av_half window.comp -DNARROW_AV)
upscaler_shader(live conv_fast conv.comp -DRGB_FAST)
upscaler_shader(live conv_poly conv.comp -DPOLYPHASE)
upscaler_shader(live conv_win_out winograd.comp -DOP=2)
upscaler_shader(live conv_win_in_half winograd.comp -DOP=0 -DNARROW)
upscaler_shader(live conv_win_mul_half winograd.comp -DOP=1 -DNARROW)
upscaler_shader(live conv_win_in_tuned winograd.comp -DOP=0 -DNARROW -DTUNED)
upscaler_shader(live conv_win_in_pair winograd.comp -DOP=0 -DNARROW -DTUNED -DREGISTER_PAIR)
upscaler_shader(live conv_win_out_tuned winograd.comp -DOP=2 -DTUNED)
upscaler_shader(live pack pack.comp)
upscaler_shader(common output4 output4.comp)
upscaler_shader(common fullscreen fullscreen.vert)
upscaler_shader(common lanczos_h lanczos.frag -DPASS=1)
upscaler_shader(common lanczos_v lanczos.frag -DPASS=2)
file(GENERATE OUTPUT "${manifest}" CONTENT "${resource_manifest}")

add_custom_command(OUTPUT "${generated}/resources.S" "${generated}/resources.cpp"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_resources.py"
            --manifest "${manifest}" --assembly "${generated}/resources.S" --index "${generated}/resources.cpp"
    DEPENDS "${manifest}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_resources.py" ${resource_dependencies}
    COMMENT "Embedding SPIR-V shaders" VERBATIM)

add_executable(upscaler
    src/main.cpp
    src/runtime.cpp
    src/config.cpp src/options.cpp src/upscale.cpp src/output_file.cpp src/png_writer.cpp src/network.cpp src/gpu.cpp src/downscale.cpp src/image_io.cpp
    src/networks/dat.cpp src/networks/rrdb.cpp
    "${generated}/resources.cpp" "${generated}/resources.S")
target_compile_features(upscaler PRIVATE cxx_std_20)
target_include_directories(upscaler PRIVATE src "${UPSCALER_STB_INCLUDE}")
target_include_directories(upscaler SYSTEM PRIVATE third_party/tomlplusplus/include)
target_compile_definitions(upscaler PRIVATE UPSCALER_VERSION="${PROJECT_VERSION}")
target_compile_options(upscaler PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-mcpu=cortex-a76>")
target_link_libraries(upscaler PRIVATE Vulkan::Vulkan ${zlib_target})
set_target_properties(upscaler PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${UPSCALER_RUNTIME_DIR}/bin")
add_dependencies(upscaler upscaler_models private_v3dv)
install(TARGETS upscaler RUNTIME DESTINATION bin)

include(CTest)
option(UPSCALER_GPU_TESTS "Run CTest Vulkan device checks on a Raspberry Pi 5" OFF)
if(BUILD_TESTING)
    add_test(NAME upscaler_version COMMAND upscaler --version)
    if(UPSCALER_GPU_TESTS)
        add_test(NAME upscaler_private_driver COMMAND upscaler --driver-info)
    endif()
endif()
