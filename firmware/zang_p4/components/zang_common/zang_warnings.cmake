# include() this after idf_component_register() in every Zang component: strict warnings on OUR code only.
target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra -Wshadow -Wno-unused-parameter -Werror=implicit-function-declaration)
