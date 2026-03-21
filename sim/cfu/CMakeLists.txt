cmake_minimum_required(VERSION 3.8)
project(cfu)

set(VIL_DIR "/opt/renode/plugins/IntegrationLibrary/src")
set(VTOP    "${CMAKE_CURRENT_SOURCE_DIR}/../../top.v")

find_package(verilator REQUIRED)

# Library sources — bus protocol + Renode agent glue.
set(RENODE_SOURCES ${VIL_DIR}/buses/cfu.cpp ${VIL_DIR}/renode_cfu.cpp)

add_library(libVtop SHARED sim_main.cpp ${RENODE_SOURCES})
target_include_directories(libVtop PRIVATE ${VIL_DIR})

# The /opt/renode v1.16.1 cfu.h is missing signal-pointer fields.
# Force-include our patched copy so its include guard blocks the
# system one.  Also pull in renode_imports.h (provides EXTERNAL_AS
# macro used by renode_cfu.cpp but not transitively included).
target_compile_options(libVtop PRIVATE
    "SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/buses/cfu.h"
    "SHELL:-include ${VIL_DIR}/renode_imports.h"
)

set_target_properties(libVtop PROPERTIES OUTPUT_NAME Vtop)

verilate(libVtop
    SOURCES        ${VTOP}
    TRACE
    VERILATOR_ARGS -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-UNOPTFLAT
)
