cmake_minimum_required(VERSION 3.20)
project(upscaler VERSION 1.0.0 LANGUAGES CXX ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Vulkan REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)
find_program(MESON_EXECUTABLE meson REQUIRED)
find_path(UPSCALER_STB_INCLUDE stb_image_write.h
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb" PATH_SUFFIXES stb REQUIRED)
find_path(TOML_INCLUDE toml++/toml.hpp
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tomlplusplus/include" REQUIRED)

set(UPSCALER_MESA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/mesa" CACHE PATH "Clean Mesa source tree or Git checkout")
set(UPSCALER_MESA_JOBS 2 CACHE STRING "Parallel jobs for the Mesa build")
set(UPSCALER_MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models/checkpoints" CACHE PATH "Local model checkpoints or packed weights")

set(product "${CMAKE_CURRENT_BINARY_DIR}/product")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "$<1:${product}>")
set(private_driver "${product}/libvulkan_broadcom.so")
set(mesa_patch "${CMAKE_CURRENT_SOURCE_DIR}/patches/mesa/0001-v3dv-upscaler-packed-fp16.patch")
set(mesa_builder "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_v3dv.py")
add_custom_command(OUTPUT "${private_driver}"
    COMMAND "${Python3_EXECUTABLE}" -B "${mesa_builder}"
        "${UPSCALER_MESA_SOURCE}" "${mesa_patch}" "${private_driver}"
        --meson "${MESON_EXECUTABLE}" --jobs "${UPSCALER_MESA_JOBS}"
    COMMAND "${CMAKE_STRIP}" --strip-unneeded "${private_driver}"
    DEPENDS "${mesa_builder}" "${mesa_patch}" "${UPSCALER_MESA_SOURCE}/VERSION"
    COMMENT "Building the private Broadcom Vulkan driver" USES_TERMINAL VERBATIM)
set_source_files_properties("${private_driver}" PROPERTIES HEADER_FILE_ONLY TRUE)

set(generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${generated}")
set(manifest "${generated}/resources.txt")

function(upscaler_model mode checkpoint reference url)
    set(packed "${UPSCALER_MODELS_DIR}/${mode}.packed.dat")
    if(NOT EXISTS "${packed}")
        set(source "${UPSCALER_MODELS_DIR}/${checkpoint}")
        if(NOT EXISTS "${source}")
            set(source "${generated}/checkpoints/${checkpoint}")
            add_custom_command(OUTPUT "${source}"
                COMMAND "${CMAKE_COMMAND}" -DURL=${url} -DOUTPUT=${source}
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchModel.cmake"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchModel.cmake"
                COMMENT "Downloading ${mode} checkpoint" VERBATIM)
        endif()
        set(packed "${generated}/${mode}.packed.dat")
        set(converter "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_${mode}.py")
        add_custom_command(OUTPUT "${packed}"
            COMMAND "${Python3_EXECUTABLE}" -B "${converter}" --checkpoint "${source}" --output "${packed}"
            DEPENDS "${source}" "${converter}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/package_model.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/${reference}"
            COMMENT "Packing ${mode} model weights" VERBATIM)
    endif()
    set(resource_manifest "${resource_manifest}${mode}/model|${packed}\n" PARENT_SCOPE)
    set(resource_dependencies "${resource_dependencies};${packed}" PARENT_SCOPE)
endfunction()

upscaler_model(anime 4x_APISR_DAT_GAN_generator.pth dat.py
    https://github.com/Kiteretsu77/APISR/releases/download/v0.3.0/4x_APISR_DAT_GAN_generator.pth)
upscaler_model(live RealESRGAN_x4plus.pth rrdbnet_arch.py
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

function(upscaler_shader mode name source)
    set(output "${generated}/${mode}/${name}.spv")
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${mode}/${source}")
    add_custom_command(OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${generated}/${mode}"
        COMMAND "${GLSLC_EXECUTABLE}" -O --target-env=vulkan1.1 -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders/common"
            -MD -MF "${output}.d" ${ARGN} "${input}" -o "${output}"
        DEPENDS "${input}"
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
upscaler_shader(anime op4_half ops.comp -DOP=4)
upscaler_shader(anime softmax_half softmax.comp)
upscaler_shader(anime window_av_half window.comp)
upscaler_shader(live conv_fast conv.comp)
upscaler_shader(live conv_poly conv.comp -DPOLYPHASE)
upscaler_shader(live conv_win_mul_half winograd.comp -DOP=1)
upscaler_shader(live conv_win_in_single winograd.comp -DOP=0)
upscaler_shader(live conv_win_in_pair winograd.comp -DOP=0 -DREGISTER_PAIR)
upscaler_shader(live conv_win_out winograd.comp -DOP=2)
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
    COMMENT "Embedding model weights and SPIR-V shaders" VERBATIM)

add_executable(upscaler
    src/main.cpp src/runtime.cpp src/config.cpp src/options.cpp src/upscale.cpp
    src/output_file.cpp src/png_writer.cpp src/network.cpp src/gpu.cpp
    src/downscale.cpp src/image_io.cpp src/networks/dat.cpp src/networks/rrdb.cpp
    "${generated}/resources.cpp" "${generated}/resources.S" "${private_driver}")
target_include_directories(upscaler PRIVATE src)
target_include_directories(upscaler SYSTEM PRIVATE "${UPSCALER_STB_INCLUDE}" "${TOML_INCLUDE}")
target_compile_definitions(upscaler PRIVATE UPSCALER_VERSION="${PROJECT_VERSION}")
target_link_libraries(upscaler PRIVATE Vulkan::Vulkan ${CMAKE_DL_LIBS})

install(TARGETS upscaler RUNTIME DESTINATION .)
install(FILES "${private_driver}" DESTINATION .)
