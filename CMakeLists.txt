cmake_minimum_required(VERSION 3.13)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for SonarLint

set(PROJECT_NAME
  distributed_hhe_ppml  
)

set(SOURCE_DIR
    src
)

set(PROJECT_SOURCE
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/examples/pktnn_examples.cpp
    ${SOURCE_DIR}/examples/hhe_pktnn_examples.cpp
    # ${SOURCE_DIR}/examples/quantization.cpp
    ${SOURCE_DIR}/util/sealhelper.cpp
    ${SOURCE_DIR}/util/pastahelper.cpp
    ${SOURCE_DIR}/util/utils.cpp
    ${SOURCE_DIR}/pasta/pasta_3_plain.cpp
    ${SOURCE_DIR}/pasta/pasta_3_seal.cpp
    ${SOURCE_DIR}/pasta/SEAL_Cipher.cpp
    configs/config.cpp
    tests/ecg_tests.cpp
    tests/mnist_tests.cpp
)

set(LIBRARIES_DIR
    libs
)

set(LIBRARIES 
    pocketnn
    aes
    keccak
)

project(${PROJECT_NAME})

# Set the CMAKE_PREFIX_PATH to the directory containing SEAL
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_SOURCE_DIR}/libs/seal)
find_package(SEAL 4.0 REQUIRED)

# for AES code
include(CheckCCompilerFlag)
function(check_c_compiler_flag_and_add flag result)
  check_c_compiler_flag("${flag}" ${result})
  if(${result})
    add_compile_options("${flag}")
  endif()
endfunction()

if(MSVC)
  add_definitions("-DOC_ENABLE_SSE2")
  message("-- Enabling SSE2 for AES")
  add_definitions("-DOC_ENABLE_AESNI")
  message("-- Enabling AES_NI")
else()
  check_c_compiler_flag_and_add(-msse2 CC_SUPPORTS_SSE2)
  if(CC_SUPPORTS_SSE2)
    add_definitions("-DOC_ENABLE_SSE2")
    message("-- Enabling SSE2 for AES")
  endif()

  check_c_compiler_flag_and_add(-maes CC_SUPPORTS_AES)
  if(CC_SUPPORTS_AES)
    add_definitions("-DOC_ENABLE_AESNI")
    message("-- Enabling AES_NI")
  else()
    add_definitions("-DOC_ENABLE_PORTABLE_AES")
  endif()
endif()

# Set this to ON if GRPC and Protobuf are installed on the system
set(USE_SYSTEM_GRPC OFF)

if(USE_SYSTEM_GRPC)
  # Find Protobuf installation
  # Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
  set(protobuf_MODULE_COMPATIBLE TRUE)
  find_package(Protobuf CONFIG REQUIRED)
  message(STATUS "Using protobuf ${Protobuf_VERSION}")

  set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
  set(_REFLECTION gRPC::grpc++_reflection)
  if(CMAKE_CROSSCOMPILING)
    find_program(_PROTOBUF_PROTOC protoc)
  else()
    set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
  endif()

  # Find gRPC installation
  # Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
  find_package(gRPC CONFIG REQUIRED)
  message(STATUS "Using gRPC ${gRPC_VERSION}")

  set(_GRPC_GRPCPP gRPC::grpc++)
  if(CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
  else()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
  endif()
else()  
  include(FetchContent)
  FetchContent_Declare(
    gRPC
    GIT_REPOSITORY https://github.com/grpc/grpc
    GIT_TAG        v1.53.0
  )
  set(FETCHCONTENT_QUIET OFF)
  FetchContent_MakeAvailable(gRPC)

  set(_PROTOBUF_LIBPROTOBUF libprotobuf)
  set(_REFLECTION grpc++_reflection)
  set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
  set(_GRPC_GRPCPP grpc++)
  if(CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
  else()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
  endif()
endif()

# Fetch and build TBB
FetchContent_Declare(
    tbb
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    GIT_TAG        v2022.0.0
)

FetchContent_MakeAvailable(tbb)

# Proto file
get_filename_component(hhe_proto "${PROJECT_SOURCE_DIR}/protos/hhe.proto" ABSOLUTE)
get_filename_component(hhe_proto_path "${hhe_proto}" PATH)

# Generated sources
set(hhe_proto_srcs "${PROJECT_SOURCE_DIR}/src/gen_code/hhe.pb.cc")
set(hhe_proto_hdrs "${PROJECT_SOURCE_DIR}/src/gen_code/hhe.pb.h")
set(hhe_grpc_srcs "${PROJECT_SOURCE_DIR}/src/gen_code/hhe.grpc.pb.cc")
set(hhe_grpc_hdrs "${PROJECT_SOURCE_DIR}/src/gen_code/hhe.grpc.pb.h")
add_custom_command(
      OUTPUT "${hhe_proto_srcs}" "${hhe_proto_hdrs}" "${hhe_grpc_srcs}" "${hhe_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${PROJECT_SOURCE_DIR}/src/gen_code"
        --cpp_out "${PROJECT_SOURCE_DIR}/src/gen_code"
        -I "${hhe_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hhe_proto}"
      DEPENDS "${hhe_proto}")

# Include generated *.pb.h files
include_directories("${PROJECT_SOURCE_DIR}/src/gen_code")

# hhe_grpc_proto
add_library(hhe_grpc_proto
  ${hhe_grpc_srcs}
  ${hhe_grpc_hdrs}
  ${hhe_proto_srcs}
  ${hhe_proto_hdrs})
target_link_libraries(hhe_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

set(ANALYST_SOURCE
    ${SOURCE_DIR}/examples/Analyst/AnalystRPC.cpp
    ${SOURCE_DIR}/examples/Analyst/Analyst.cpp
    ${SOURCE_DIR}/examples/Analyst/CSPServiceAnalystClient.cpp
    ${SOURCE_DIR}/util/sealhelper.cpp
    ${SOURCE_DIR}/util/pastahelper.cpp
    ${SOURCE_DIR}/util/utils.cpp
    ${SOURCE_DIR}/pasta/pasta_3_plain.cpp
    ${SOURCE_DIR}/pasta/pasta_3_seal.cpp
    ${SOURCE_DIR}/pasta/SEAL_Cipher.cpp
    configs/config.cpp
)

set(USER_SOURCE
    ${PROJECT_SOURCE_DIR}/src/examples/User/AnalystServiceUserClient.cpp
    ${PROJECT_SOURCE_DIR}/src/examples/User/CSPServiceUserClient.cpp
    ${PROJECT_SOURCE_DIR}/src/examples/User/User.cpp
    ${PROJECT_SOURCE_DIR}/src/examples/User/UserRPC.cpp
    ${PROJECT_SOURCE_DIR}/src/util/sealhelper.cpp
    ${PROJECT_SOURCE_DIR}/src/util/pastahelper.cpp
    ${PROJECT_SOURCE_DIR}/src/util/utils.cpp
    ${PROJECT_SOURCE_DIR}/src/pasta/pasta_3_plain.cpp
    ${PROJECT_SOURCE_DIR}/src/pasta/pasta_3_seal.cpp
    ${PROJECT_SOURCE_DIR}/src/pasta/SEAL_Cipher.cpp
    configs/config.cpp
)

set(CSP_SOURCE
    ${SOURCE_DIR}/examples/CSP/CSP.cpp
    ${SOURCE_DIR}/examples/CSP/CSPRPC.cpp
    ${SOURCE_DIR}/examples/CSP/AnalystServiceCSPClient.cpp
    ${SOURCE_DIR}/util/sealhelper.cpp
    ${SOURCE_DIR}/util/pastahelper.cpp
    ${SOURCE_DIR}/util/utils.cpp
    ${SOURCE_DIR}/pasta/pasta_3_plain.cpp
    ${SOURCE_DIR}/pasta/pasta_3_seal.cpp
    ${SOURCE_DIR}/pasta/SEAL_Cipher.cpp
    configs/config.cpp
)

add_executable(single_hhe_ppml ${PROJECT_SOURCE})

add_executable(analyst ${ANALYST_SOURCE})
add_executable(user ${USER_SOURCE}) 
add_executable(csp ${CSP_SOURCE})

foreach(LIBRARY ${LIBRARIES})
    add_subdirectory("${LIBRARIES_DIR}/${LIBRARY}")
    message(STATUS "Libraries included: ${LIBRARIES_DIR}/${LIBRARY}")
endforeach(LIBRARY)

target_link_libraries(single_hhe_ppml ${LIBRARIES})
target_link_libraries(single_hhe_ppml SEAL::seal)

target_link_libraries(csp
    SEAL::seal
    keccak
    aes
    pocketnn
    hhe_grpc_proto
    TBB::tbb
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})

target_link_libraries(analyst
    SEAL::seal
    keccak
    aes
    pocketnn
    hhe_grpc_proto
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})

target_link_libraries(user
    SEAL::seal
    keccak
    aes
    pocketnn
    hhe_grpc_proto
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})



