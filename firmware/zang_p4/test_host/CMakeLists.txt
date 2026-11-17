# Host unit tests for the pure-C firmware modules. Plain CMake + gcc, no ESP-IDF needed:
#   cmake -S firmware/zang_p4/test_host -B firmware/zang_p4/test_host/build && cmake --build ... && ctest --test-dir ...
cmake_minimum_required(VERSION 3.16)
project(zang_test_host C)
set(CMAKE_C_STANDARD 11)
set(C ${CMAKE_CURRENT_SOURCE_DIR}/../components)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(VEC_JSON ${C}/zlink/test/zlink_vectors.json)
set(VEC_H ${CMAKE_CURRENT_BINARY_DIR}/zlink_vectors.h)
add_custom_command(OUTPUT ${VEC_H}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_vectors_h.py ${VEC_JSON} ${VEC_H}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_vectors_h.py
    COMMENT "zlink_vectors.json -> zlink_vectors.h")
if(EXISTS ${VEC_JSON})
    set_property(SOURCE ${VEC_H} APPEND PROPERTY OBJECT_DEPENDS ${VEC_JSON})
    add_custom_target(vectors_dep DEPENDS ${VEC_JSON})
endif()

add_executable(zang_tests
    test_main.c ${VEC_H}
    ${C}/zlink/zlink.c
    ${C}/servo_bus/ds_proto.c
    ${C}/motion/motion_math.c
    ${C}/robot_def/robot_joints.c
    ${C}/app/app_sm.c
    ${C}/audio/wav.c)
target_include_directories(zang_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    ${C}/zlink/include ${C}/servo_bus/include ${C}/motion/include ${C}/robot_def/include
    ${C}/app/include ${C}/audio ${C}/zang_common/include)
target_compile_options(zang_tests PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter -fsanitize=address,undefined -g)
target_link_options(zang_tests PRIVATE -fsanitize=address,undefined)
target_link_libraries(zang_tests m)

enable_testing()
add_test(NAME unit COMMAND zang_tests)
add_test(NAME robot_joints_up_to_date
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/gen_robot_joints.py --check)
add_test(NAME board_pins_up_to_date
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/gen_board_pins.py --check)

# ---- face renderer: goldens from ai_stack/zang_ai/face/render.py (components/display/test/faces) ----
set(FACE_JSON ${C}/display/test/faces/goldens.json)
set(FACE_H ${CMAKE_CURRENT_BINARY_DIR}/faces_goldens.h)
add_custom_command(OUTPUT ${FACE_H}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_faces_h.py ${FACE_JSON} ${FACE_H}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_faces_h.py ${FACE_JSON}
    COMMENT "goldens.json -> faces_goldens.h")
add_executable(face_tests face_test.c ${FACE_H} ${C}/display/face_sdf.c ${C}/display/faces_lib.c)
target_include_directories(face_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${C}/display/include)
target_compile_definitions(face_tests PRIVATE GOLDEN_DIR="${C}/display/test/faces")
target_compile_options(face_tests PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter -fsanitize=address,undefined -g)
target_link_options(face_tests PRIVATE -fsanitize=address,undefined)
target_link_libraries(face_tests m)
add_test(NAME faces COMMAND face_tests)
add_test(NAME faces_lib_up_to_date
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/gen_faces.py --check)

add_executable(engine_tests engine_test.c ${C}/display/face_engine.c ${C}/display/face_sdf.c ${C}/display/faces_lib.c)
target_include_directories(engine_tests PRIVATE ${C}/display/include)
target_compile_options(engine_tests PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter -fsanitize=address,undefined -g)
target_link_options(engine_tests PRIVATE -fsanitize=address,undefined)
target_link_libraries(engine_tests m)
add_test(NAME face_engine COMMAND engine_tests)

# render-time benchmark (no sanitizers, -O2); prints host ms per frame
add_executable(face_bench face_bench.c ${FACE_H} ${C}/display/face_sdf.c ${C}/display/faces_lib.c)
target_include_directories(face_bench PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${C}/display/include)
target_compile_options(face_bench PRIVATE -O2)
target_link_libraries(face_bench m)
add_test(NAME face_bench COMMAND face_bench)

add_executable(media_tests media_test.c ${C}/display/canvas.c ${C}/display/img_asm.c)
target_include_directories(media_tests PRIVATE ${C}/display/include ${C}/display)
target_compile_options(media_tests PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter -fsanitize=address,undefined -g)
target_link_options(media_tests PRIVATE -fsanitize=address,undefined)
add_test(NAME media COMMAND media_tests)
