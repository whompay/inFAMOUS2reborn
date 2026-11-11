cmake_minimum_required(VERSION 3.20)
project(inFamous2Recomp
    VERSION 0.1.0
    LANGUAGES C CXX
    DESCRIPTION "inFamous 2 (BCES01143) static recompilation"
)

# ---------------------------------------------------------------------------
# Standards
# ---------------------------------------------------------------------------
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Whole-program profile-guided optimization. GEN instruments the lifted code,
# runtime and renderer; USE feeds the merged profile back into Clang for code
# layout, inlining, branch ordering and register-allocation decisions.
set(PORT_PGO "OFF" CACHE STRING "PGO mode: OFF, GEN or USE")
set_property(CACHE PORT_PGO PROPERTY STRINGS OFF GEN USE)
set(PORT_PGO_PROFILE "" CACHE FILEPATH "Merged .profdata file for PORT_PGO=USE")
if(NOT PORT_PGO MATCHES "^(OFF|GEN|USE)$")
    message(FATAL_ERROR "PORT_PGO must be OFF, GEN or USE (got '${PORT_PGO}')")
endif()
if(PORT_PGO STREQUAL "GEN")
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "PORT_PGO=GEN currently requires Clang")
    endif()
    add_compile_options(-fprofile-generate "SHELL:-mllvm -disable-vp=true")
    add_compile_definitions(PORT_PGO_GEN=1)
    get_filename_component(_pgo_llvm_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    file(GLOB _pgo_rt "${_pgo_llvm_bin}/../lib/clang/*/lib/windows/clang_rt.profile-x86_64.lib")
    if(NOT _pgo_rt)
        message(FATAL_ERROR "PORT_PGO=GEN: clang profile runtime was not found next to ${CMAKE_CXX_COMPILER}")
    endif()
    list(GET _pgo_rt 0 _pgo_rt)
    get_filename_component(_pgo_rt_dir "${_pgo_rt}" DIRECTORY)
    add_link_options("${_pgo_rt}" "/LIBPATH:${_pgo_rt_dir}")
    message(STATUS "PGO:             GEN (${_pgo_rt})")
elseif(PORT_PGO STREQUAL "USE")
    file(TO_CMAKE_PATH "${PORT_PGO_PROFILE}" PORT_PGO_PROFILE)
    if(NOT EXISTS "${PORT_PGO_PROFILE}")
        message(FATAL_ERROR "PORT_PGO=USE requires PORT_PGO_PROFILE=<merged.profdata>")
    endif()
    add_compile_options("-fprofile-use=${PORT_PGO_PROFILE}"
        -Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date
        -Wno-profile-instr-missing -Wno-backend-plugin)
    message(STATUS "PGO:             USE (${PORT_PGO_PROFILE})")
else()
    message(STATUS "PGO:             OFF")
endif()

# v170P (perf): this build tree's cache carries CMAKE_C_FLAGS_RELEASE="" and
# CMAKE_CXX_FLAGS_RELEASE="", and clang-cl with no /O flag compiles at /Od. The
# runtime library already forces -O2 -msse4.1 for itself, but this target did
# not: the 270 MB PPU lift, main.cpp and the boot scaffold were UNOPTIMISED and
# the lifted SPU sources got -O2 without -msse4.1, so spu_helpers.h fell back to
# its byte-loop shufb/rotqby/cwd forms and libm fmaf() (playbook 2.9, 11.2).
#   PORT_OPT=OFF       -> previous flags (no optimisation outside -O2 for SPU).
#   PORT_HOST_ISA=AUTO -> use AVX2/FMA/BMI only when the build host supports all of them.
#   PORT_HOST_ISA=OFF  -> SSE4.1 only (portable fallback for older x86-64 CPUs).
# Change with: cmake -S . -B build-fast -DPORT_OPT=OFF (cached), then build.cmd.
option(PORT_OPT "v170P: optimise the port target (-O2; lifted PPU -O1)" ON)
# Fresh player builds are made on the machine that will run them. Detect the
# optional ISA there so modern CPUs keep the fast path while older x86-64 CPUs
# get the SSE4.1 baseline instead of an illegal-instruction crash. Explicit ON
# and OFF remain available for reproducible packages and cross-compilation.
set(PORT_HOST_ISA "AUTO" CACHE STRING
    "Host ISA mode: AUTO, ON (AVX2/FMA/BMI/BMI2/LZCNT), or OFF (SSE4.1 baseline)")
set_property(CACHE PORT_HOST_ISA PROPERTY STRINGS AUTO ON OFF)
if(NOT PORT_HOST_ISA MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "PORT_HOST_ISA must be AUTO, ON or OFF (got '${PORT_HOST_ISA}')")
endif()
set(PORT_HOST_ISA_EFFECTIVE OFF)
if(PORT_HOST_ISA STREQUAL "ON")
    set(PORT_HOST_ISA_EFFECTIVE ON)
elseif(PORT_HOST_ISA STREQUAL "AUTO" AND
       CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|x64|X64|amd64" AND
       NOT CMAKE_CROSSCOMPILING)
    include(CheckCXXSourceRuns)
    set(CMAKE_REQUIRED_QUIET TRUE)
    # clang-cl's __builtin_cpu_supports references compiler-rt CPU-model
    # globals that are not linked into a normal MSVC executable. Query CPUID
    # directly so this configure check stays a self-contained baseline binary.
    check_cxx_source_runs("#include <intrin.h>
    int main() {
        int r[4] = {};
        __cpuid(r, 0);
        if (r[0] < 7) return 1;
        __cpuidex(r, 1, 0);
        const int fma = (r[2] & (1 << 12)) != 0;
        const int osxsave = (r[2] & (1 << 27)) != 0;
        const int avx = (r[2] & (1 << 28)) != 0;
        if (!fma || !osxsave || !avx || (_xgetbv(0) & 6) != 6) return 1;
        __cpuidex(r, 7, 0);
        const int bmi = (r[1] & (1 << 3)) != 0;
        const int avx2 = (r[1] & (1 << 5)) != 0;
        const int bmi2 = (r[1] & (1 << 8)) != 0;
        __cpuid(r, (int)0x80000000u);
        if ((unsigned)r[0] < 0x80000001u) return 1;
        __cpuidex(r, (int)0x80000001u, 0);
        const int lzcnt = (r[2] & (1 << 5)) != 0;
        return avx2 && bmi && bmi2 && lzcnt ? 0 : 1;
    }" PORT_HOST_ISA_DETECTED)
    unset(CMAKE_REQUIRED_QUIET)
    if(PORT_HOST_ISA_DETECTED)
        set(PORT_HOST_ISA_EFFECTIVE ON)
    endif()
endif()
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|x64|X64|amd64")
    set(PORT_ISA_FLAGS "")
elseif(PORT_HOST_ISA_EFFECTIVE)
    set(PORT_ISA_FLAGS -msse4.1 -mavx2 -mfma -mbmi -mbmi2 -mlzcnt)
else()
    set(PORT_ISA_FLAGS -msse4.1)
endif()
message(STATUS "v170P port flags: PORT_OPT=${PORT_OPT} PORT_HOST_ISA=${PORT_HOST_ISA} -> ${PORT_HOST_ISA_EFFECTIVE} (${PORT_ISA_FLAGS})")

# Prefer Ninja if available
if(NOT CMAKE_GENERATOR MATCHES "Ninja" AND NOT DEFINED ENV{CMAKE_GENERATOR})
    message(STATUS "Tip: use -G Ninja for faster builds")
endif()

# ---------------------------------------------------------------------------
# The two directories this project is built from
# ---------------------------------------------------------------------------
# PS3RECOMP_DIR is the toolkit checkout: include/, runtime/, libs/, tools/.
# RECOMP_DIR is where the lifter's output for THIS game lives. They are
# separate on purpose -- a port keeps its generated code in its own tree and
# points at whatever toolkit checkout it is building against:
#
#   cmake -B build -G Ninja \
#         -DPS3RECOMP_DIR=/path/to/ps3recomp \
#         -DRECOMP_DIR=/path/to/my_game/recompiled
#
if(NOT DEFINED PS3RECOMP_DIR)
    # This port and the patched toolkit are sibling directories.
    get_filename_component(PS3RECOMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ps3recomp" ABSOLUTE)
endif()
get_filename_component(PS3RECOMP_DIR "${PS3RECOMP_DIR}" ABSOLUTE)

set(RECOMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/recompiled" CACHE PATH
    "Directory holding the lifter's output: ppu_recomp.h and the lifted sources")
get_filename_component(RECOMP_DIR "${RECOMP_DIR}" ABSOLUTE)

message(STATUS "ps3recomp SDK:   ${PS3RECOMP_DIR}")
message(STATUS "Lifted output:   ${RECOMP_DIR}")

if(NOT EXISTS "${RECOMP_DIR}/ppu_recomp.h")
    message(FATAL_ERROR
        "No ppu_recomp.h in ${RECOMP_DIR}.\n"
        "That header and the lifted sources beside it are what "
        "tools/ppu_lifter.py writes. Run the lifter, drop its output there, "
        "or point -DRECOMP_DIR= at wherever it already is.")
endif()

# The lifted translation units. A game's lift is hundreds of generated files,
# so they are globbed rather than listed; re-run CMake after a fresh lift.
file(GLOB RECOMP_SOURCES CONFIGURE_DEPENDS
    "${RECOMP_DIR}/*.c"
    "${RECOMP_DIR}/*.cpp"
)
if(NOT RECOMP_SOURCES)
    message(FATAL_ERROR
        "${RECOMP_DIR} has a ppu_recomp.h but no .c or .cpp beside it. The "
        "lifted functions and the function table live in those; without them "
        "there is nothing to run.")
endif()
list(LENGTH RECOMP_SOURCES RECOMP_SOURCE_COUNT)
message(STATUS "Lifted sources:  ${RECOMP_SOURCE_COUNT} file(s)")

# ---------------------------------------------------------------------------
# ps3recomp runtime library
# ---------------------------------------------------------------------------
# Built from source as a subdirectory rather than imported as a prebuilt
# archive. A static library carries none of its own link requirements, so an
# imported one arrives without SDL2, pthreads, the Metal frameworks or the
# Winsock/XInput imports -- the link then fails on every platform for a
# different reason. add_subdirectory hands all of that over with the target.
add_subdirectory("${PS3RECOMP_DIR}" "${CMAKE_BINARY_DIR}/ps3recomp_sdk"
                 EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# The HLE NID table
# ---------------------------------------------------------------------------
# Maps each firmware import NID onto the handler in the runtime that answers
# it. A port may keep a hand-curated one beside its lift; otherwise generate
# the canonical every-module table, which is what the toolkit's own boot
# harness uses.
if(EXISTS "${RECOMP_DIR}/ppu_hle_nids.cpp")
    set(HLE_NID_TABLE "${RECOMP_DIR}/ppu_hle_nids.cpp")
    message(STATUS "HLE NID table:   ${HLE_NID_TABLE}")
else()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(HLE_NID_TABLE "${CMAKE_CURRENT_BINARY_DIR}/ppu_hle_nids.cpp")
    add_custom_command(
        OUTPUT "${HLE_NID_TABLE}"
        COMMAND "${Python3_EXECUTABLE}"
                "${PS3RECOMP_DIR}/tools/gen_hle_nids.py"
                --out "${HLE_NID_TABLE}" --all
        DEPENDS "${PS3RECOMP_DIR}/tools/gen_hle_nids.py"
                "${PS3RECOMP_DIR}/tools/nid_database.py"
        COMMENT "Generating the HLE NID -> handler table")
    message(STATUS "HLE NID table:   generated (every module)")
endif()

# ---------------------------------------------------------------------------
# Lifted PPU register caching (opt-in performance experiment)
# ---------------------------------------------------------------------------
# ppu_regcache.py keeps frequently used guest registers in native locals and
# synchronizes them around calls/returns.  A semicolon-separated basename list
# makes the experiment selective: recompiling one hot ~40 MB lifted TU is much
# quicker than rebuilding the entire lift, and an empty list preserves the
# original sources byte-for-byte.
set(PORT_PPU_REGCACHE_ONLY "" CACHE STRING
    "Lifted PPU TU basenames to register-cache (for example ppu_recomp_001)")
set(PORT_PPU_REGCACHE_MINREF "4" CACHE STRING
    "Minimum references before a guest PPU register is cached")
set(PORT_PPU_REGCACHE_SCRIPT "${PS3RECOMP_DIR}/tools/ppu_regcache.py" CACHE FILEPATH
    "PPU register-cache transformation script")
set(PORT_PPU_REGCACHE_ARGS "" CACHE STRING
    "Additional arguments passed to the PPU register-cache script")
if(PORT_PPU_REGCACHE_ONLY)
    if(NOT PORT_PPU_REGCACHE_MINREF MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "PORT_PPU_REGCACHE_MINREF must be a positive integer")
    endif()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(_ppu_rc_script "${PORT_PPU_REGCACHE_SCRIPT}")
    if(NOT EXISTS "${_ppu_rc_script}")
        message(FATAL_ERROR "PPU register-cache script does not exist: ${_ppu_rc_script}")
    endif()
    set(_ppu_rc_dir "${CMAKE_CURRENT_BINARY_DIR}/recompiled_rc")
    file(MAKE_DIRECTORY "${_ppu_rc_dir}")
    set(_ppu_rc_sources "")
    set(_ppu_rc_matched "")

    foreach(_src IN LISTS RECOMP_SOURCES)
        get_filename_component(_name "${_src}" NAME)
        get_filename_component(_stem "${_src}" NAME_WE)
        if(_stem IN_LIST PORT_PPU_REGCACHE_ONLY)
            if(NOT _name MATCHES "^ppu_recomp_[0-9]+\\.(c|cpp)$")
                message(FATAL_ERROR
                    "PORT_PPU_REGCACHE_ONLY entry ${_stem} is not a lifted PPU TU")
            endif()
            set(_out "${_ppu_rc_dir}/${_name}")
            add_custom_command(
                OUTPUT "${_out}"
                COMMAND "${CMAKE_COMMAND}" -E env
                        "PPU_RC_MINREF=${PORT_PPU_REGCACHE_MINREF}"
                        "${Python3_EXECUTABLE}" "${_ppu_rc_script}"
                        "${_src}" "${_out}" ${PORT_PPU_REGCACHE_ARGS} --stats
                DEPENDS "${_src}" "${_ppu_rc_script}"
                COMMENT "PPU register caching: ${_name}"
                VERBATIM)
            list(APPEND _ppu_rc_sources "${_out}")
            list(APPEND _ppu_rc_matched "${_stem}")
        else()
            list(APPEND _ppu_rc_sources "${_src}")
        endif()
    endforeach()

    foreach(_requested IN LISTS PORT_PPU_REGCACHE_ONLY)
        if(NOT _requested IN_LIST _ppu_rc_matched)
            message(FATAL_ERROR
                "PORT_PPU_REGCACHE_ONLY requested ${_requested}, but no matching lifted source exists")
        endif()
    endforeach()
    set(RECOMP_SOURCES ${_ppu_rc_sources})
    message(STATUS "Lifted PPU RC:   ${PORT_PPU_REGCACHE_ONLY} (minref ${PORT_PPU_REGCACHE_MINREF}, script ${_ppu_rc_script}, args ${PORT_PPU_REGCACHE_ARGS})")
else()
    message(STATUS "Lifted PPU RC:   disabled")
endif()

# ---------------------------------------------------------------------------
# The game executable
# ---------------------------------------------------------------------------
# The PPU boot scaffold is compiled here rather than taken from the runtime
# library: every one of these files #includes the lifter's per-game
# ppu_recomp.h, so they can only be built against a particular lift.
set(PS3RECOMP_SCAFFOLD
    "${PS3RECOMP_DIR}/runtime/ppu/ppu_loader.cpp"
    "${PS3RECOMP_DIR}/runtime/ppu/ppu_hle.cpp"
    "${PS3RECOMP_DIR}/runtime/ppu/ppu_fs.cpp"
    "${PS3RECOMP_DIR}/runtime/ppu/ppu_sysprx.cpp"
)

# ---------------------------------------------------------------------------
# Lifted SPU images (optional)
# ---------------------------------------------------------------------------
# A title that runs SPU work gets its images lifted into <SPU_DIR>/<image>/ by
# tools/spu_lifter.py, with a registration TU generated by
# tools/build_spu_workloads.py. cellSpurs dispatches them by content
# fingerprint, so the generated table is keyed to the images THIS build ships:
# a table generated from another release registers signatures that can never
# match, which is worse than shipping none. Both the table and the job-module
# registry self-register from constructors and are therefore optional -- a tree
# that has not lifted its SPU images yet configures and links without them.
set(SPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/spu" CACHE PATH
    "Directory holding the lifted SPU images (<SPU_DIR>/<image>/spu_recomp.c)")
option(PORT_ENABLE_SPU "Compile and register the title's lifted SPU images" ON)
option(PORT_SKIP_PREBUILT_RC "Omit the unused oversized static SPU register-cache twin" OFF)

set(SPU_SOURCES "")
set(SPU_PLUG_FALLBACK "")
if(PORT_ENABLE_SPU)
    file(GLOB SPU_LIFTED CONFIGURE_DEPENDS
        "${SPU_DIR}/*/spu_recomp.c"
        "${SPU_DIR}/*/spu_recomp_rc.c")
    if(PORT_SKIP_PREBUILT_RC)
        # WWS plugin RC functions are referenced by its dispatcher and must
        # remain linked. Only this static image's twin is self-contained;
        # TASKWAIT FPS tests run with SPU_REGCACHE=0.
        list(FILTER SPU_LIFTED EXCLUDE REGEX "/spu_0001_at_0085E400/spu_recomp_rc\\.c$")
    endif()
    if(SPU_LIFTED)
        list(LENGTH SPU_LIFTED SPU_IMAGE_COUNT)
        message(STATUS "Lifted SPU:      ${SPU_IMAGE_COUNT} image(s)")
        # The musttail indirect-branch fast path. Excluded from the runtime
        # library on purpose (see the SDK root CMakeLists): it must be built
        # here, by clang, or the tail call degrades to a plain call and a guest
        # loop that branches indirectly walks the thread stack off its guard
        # page a few thousand iterations in.
        list(APPEND SPU_SOURCES ${SPU_LIFTED}
             "${PS3RECOMP_DIR}/runtime/spu/spu_dispatch_mt.c")
        # Generated beside the lift; each is present only once its generator
        # has been run, so each is included only if it is actually here.
        foreach(gen IN ITEMS spu_workloads.c pm_wwsjob_register.c raw_jobs_register.c jobmods_register.c taskset_policy_bytes.c)
            if(EXISTS "${SPU_DIR}/${gen}")
                list(APPEND SPU_SOURCES "${SPU_DIR}/${gen}")
            else()
                message(STATUS "  (no ${gen} -- not registered)")
            endif()
        endforeach()
    else()
        message(STATUS "Lifted SPU:      none found in ${SPU_DIR}")
    endif()
endif()

# The patched interpreter can hand control back to relocatable WWS plugin
# lifts. A fresh title has no plugin corpus yet, but the runtime still calls
# the resolver; provide a null resolver until spu/wwsplug/spu_recomp.c exists.
if(NOT EXISTS "${SPU_DIR}/wwsplug/spu_recomp.c")
    set(SPU_PLUG_FALLBACK "${CMAKE_CURRENT_SOURCE_DIR}/spu_plug_stub.c")
endif()

add_executable(${PROJECT_NAME}
    main.cpp
    stubs.cpp
    ${PS3RECOMP_SCAFFOLD}
    ${HLE_NID_TABLE}
    ${RECOMP_SOURCES}
    ${SPU_SOURCES}
    ${SPU_PLUG_FALLBACK}
)

# Keep a playable/running executable untouched while linking an experimental
# build from the same object files (Windows cannot overwrite a running .exe).
set(PORT_OUTPUT_NAME "" CACHE STRING "Optional alternate executable basename")
if(PORT_OUTPUT_NAME)
    set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "${PORT_OUTPUT_NAME}")
endif()

# The lifted SPU sources use bare #include "spu_context.h". Scope that include
# directory (and their flags) to those TUs alone: adding it to the whole target
# changes the compile command of every lifted PPU TU, and those are large enough
# that the needless rebuild is measured in tens of minutes.
#   SPU_USE_DISPATCH_MT -- route SPU_IB_DISPATCH through the musttail path.
#   -O2                 -- the target-wide -O1 exists for the huge PPU TUs; the
#                          SPU sources are small and decode-hot, so they get
#                          full optimization. The last -O on the line wins.
if(SPU_SOURCES)
    set_source_files_properties(${SPU_SOURCES} PROPERTIES
        INCLUDE_DIRECTORIES "${PS3RECOMP_DIR}/runtime/spu"
        COMPILE_DEFINITIONS SPU_USE_DISPATCH_MT)
    if(NOT MSVC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(PORT_OPT)
            # v170P: the ISA flags the helpers were written for (lifted SPU code in
            # the inFamous port: -O2 -msse4.1 -mavx2 -mfma, default contraction --
            # spu_helpers_vec.h v130 fi matches the contracted reference).
            set_source_files_properties(${SPU_SOURCES} PROPERTIES COMPILE_OPTIONS "-O2;${PORT_ISA_FLAGS}")
        else()
            set_source_files_properties(${SPU_SOURCES} PROPERTIES COMPILE_OPTIONS "-O2")
        endif()
    endif()
endif()

# F181 (perf): build-time register-cached twins of the lifted SPU images.
# The frame limiter in the world is SPU job code (w2 "SpuLight" lanes ~3 cores busy, "lifted+dispatch"
# 93%, the main thread waits ~80% of each frame for it). The lifter writes every SPU op through
# ctx->gpr in memory; ps3recomp/tools/spu_regcache.py (v95) keeps the registers in host locals and
# leaves ctx exact at every call/channel/transfer/return (offline: jobmod_6c2ac7b491f5 -35% x86
# instructions, -28% memory ops). ON: every image under SPU_DIR WITHOUT its own spu_recomp_rc.c gets
# a twin generated into <build>/spu_rc/<image>/ by tools/gen_spu_rc.py (regenerated whenever the
# lift or the pass changes, so it cannot go stale); each twin registers (original, rc) pairs and
# the runtime serves them unless SPU_REGCACHE=0. OFF (default): build unchanged.
#   cmake -S . -B <build> -DPORT_SPU_RC_GEN=ON
option(PORT_SPU_RC_GEN "F181: generate register-cached twins of every lifted SPU image (build tree)" OFF)
# Q (2026-10-16): PORT_SPU_RC_ONLY=<img;img;..> limits the generated twins to these image directories
# (e.g. the w2 batch job modules the native sampler ranks hottest). Empty (default) = every image.
set(PORT_SPU_RC_ONLY "" CACHE STRING "Q: only these spu/<image> dirs get generated rc twins (empty = all)")
# The largest batch-job RC function has enough live vectors that clang-cl's
# -O2 (LLVM -O3) spills heavily.  Permit a per-image -O1 (LLVM -Os) probe;
# empty by default so normal and existing RC builds keep their flags.
set(PORT_SPU_RC_O1_ONLY "" CACHE STRING "SPU: RC image dirs to compile at -O1 instead of -O2 (empty = none)")
option(PORT_SPU_RC_XFLOAT_FIXED "SPU: compile RC twins with fixed xfloat-on limits (original fallback when off)" OFF)
if(PORT_SPU_RC_GEN AND SPU_LIFTED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(F181_RC_TOOL "${PS3RECOMP_DIR}/tools/spu_regcache.py")
    set(F181_RC_GEN "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_spu_rc.py")
    set(F181_RC_SOURCES "")
    foreach(f181_src IN LISTS SPU_LIFTED)
        get_filename_component(f181_name "${f181_src}" NAME)
        get_filename_component(f181_dir "${f181_src}" DIRECTORY)
        get_filename_component(f181_img "${f181_dir}" NAME)
        set(f181_pick TRUE)
        if(PORT_SPU_RC_ONLY)
            list(FIND PORT_SPU_RC_ONLY "${f181_img}" f181_idx)
            if(f181_idx LESS 0)
                set(f181_pick FALSE)
            endif()
        endif()
        if(f181_pick AND f181_name STREQUAL "spu_recomp.c" AND NOT EXISTS "${f181_dir}/spu_recomp_rc.c")
            set(f181_out "${CMAKE_CURRENT_BINARY_DIR}/spu_rc/${f181_img}/spu_recomp_rc.c")
            add_custom_command(OUTPUT "${f181_out}"
                COMMAND "${Python3_EXECUTABLE}" "${F181_RC_GEN}" "${F181_RC_TOOL}" "${f181_src}" "${f181_out}"
                DEPENDS "${f181_src}" "${F181_RC_TOOL}" "${F181_RC_GEN}"
                COMMENT "F181 register-cached SPU twin: ${f181_img}"
                VERBATIM)
            list(APPEND F181_RC_SOURCES "${f181_out}")
        endif()
    endforeach()
    list(LENGTH F181_RC_SOURCES F181_RC_COUNT)
    message(STATUS "F181 PORT_SPU_RC_GEN: ${F181_RC_COUNT} generated register-cached SPU twin(s)")
    if(F181_RC_SOURCES)
        target_sources(${PROJECT_NAME} PRIVATE ${F181_RC_SOURCES})
        if(PORT_SPU_RC_XFLOAT_FIXED)
            set_source_files_properties("${PS3RECOMP_DIR}/runtime/spu/spu_channels.c"
                TARGET_DIRECTORY ps3recomp_runtime PROPERTIES COMPILE_DEFINITIONS SPU_RC_XFLOAT_FIXED_REGISTRY)
        endif()
        if(PORT_OPT)
            set(F181_RC_OPTS "-O2;${PORT_ISA_FLAGS}")
        else()
            set(F181_RC_OPTS "-O2")
        endif()
        set(f181_rc_defs SPU_USE_DISPATCH_MT)
        if(PORT_SPU_RC_XFLOAT_FIXED)
            list(APPEND f181_rc_defs SPU_RC_XFLOAT_FIXED)
        endif()
        set_source_files_properties(${F181_RC_SOURCES} PROPERTIES
            INCLUDE_DIRECTORIES "${PS3RECOMP_DIR}/runtime/spu"
            COMPILE_DEFINITIONS "${f181_rc_defs}"
            COMPILE_OPTIONS "${F181_RC_OPTS}")
        foreach(f181_rc_src IN LISTS F181_RC_SOURCES)
            get_filename_component(f181_rc_dir "${f181_rc_src}" DIRECTORY)
            get_filename_component(f181_rc_img "${f181_rc_dir}" NAME)
            list(FIND PORT_SPU_RC_O1_ONLY "${f181_rc_img}" f181_rc_o1_idx)
            if(NOT f181_rc_o1_idx LESS 0)
                if(PORT_OPT)
                    set(f181_rc_o1_opts "-O1;${PORT_ISA_FLAGS}")
                else()
                    set(f181_rc_o1_opts "-O1")
                endif()
                set_source_files_properties("${f181_rc_src}" PROPERTIES COMPILE_OPTIONS "${f181_rc_o1_opts}")
                message(STATUS "SPU RC -O1 probe: ${f181_rc_img}")
            endif()
        endforeach()
    endif()
endif()

# app.rc carries the icon, the version block and the manifest, and needs a
# resource compiler. Listing it as a source on a host that has none is not a
# link error, it is a "no rule to make" for a file nothing can consume.
if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/app.rc")
    target_sources(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/app.rc")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    "${RECOMP_DIR}"                            # the generated ppu_recomp.h
    "${PS3RECOMP_DIR}/include"
    "${PS3RECOMP_DIR}/runtime/ppu"
    "${PS3RECOMP_DIR}/runtime/spu"
    "${PS3RECOMP_DIR}/runtime/memory"
    "${PS3RECOMP_DIR}/runtime/syscalls"
    "${PS3RECOMP_DIR}/runtime/platform"        # win32_compat.h and friends
    "${PS3RECOMP_DIR}/libs/video"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ps3recomp_runtime
)

# The runtime is C++, and on Apple also Objective-C. Letting CMake infer the
# link language from the generated C would leave the C++ standard library out.
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# Windows system import libraries. dbghelp symbolizes crash-handler addresses
# against the PDB, user32 and gdi32 are what the backend's window and its GDI
# present are made of, and winmm is timeBeginPeriod. None of the four names
# anything a POSIX linker can find; the D3D12 backend's own imports come in
# through the #pragma comment(lib) in libs/video/rsx_d3d12_backend.c.
# v170P: a link map so PS3_PROF_MAIN [PROF-FN] RVAs can be resolved
# (build-fast/inFamous2Recomp.map; func_XXXXXXXX = lifted guest code).
# /OPT:NOICF: optimised objects now carry function-level COMDATs; keep every
# lifted function a distinct address (as in the previous /Od build) instead of
# letting lld fold byte-identical tail-entry copies.
if(MSVC)
    target_link_options(${PROJECT_NAME} PRIVATE "/MAP" "/OPT:NOICF")
endif()
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        dbghelp
        user32
        gdi32
        winmm
        dxguid
    )
endif()

# ---------------------------------------------------------------------------
# Compiler-specific flags
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall -Wextra -Wno-unused-parameter
        -fno-strict-aliasing
    )
    if(PORT_OPT)
        # v170P: explicit, because the per-config defaults are empty in this tree.
        # The lifted PPU sources override this with -O1 below (compile time).
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release,RelWithDebInfo>:-O2>
            ${PORT_ISA_FLAGS}
        )
        # No FP contraction in the C++ TUs (scaffold, main, lifted PPU) now that FMA
        # exists; the C TUs are the lifted SPU sources, which keep the default.
        if(MSVC)
            target_compile_options(${PROJECT_NAME} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/clang:-ffp-contract=off>)
        else()
            target_compile_options(${PROJECT_NAME} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
        endif()
    endif()
elseif(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /W3 /wd4100 /wd4244 /wd4267
        /Zc:__cplusplus         # report correct __cplusplus value
        # ppu_recomp.h defines PPU_FENCE, the lowering of the guest's
        # sync/lwsync/eieio/isync, and in C that is <stdatomic.h>. The lifter
        # writes C++ by default, but --single-file writes ppu_recomp.c, and a
        # port that took it would not compile without this. The runtime's own
        # build already carries the same flag for the same reason.
        /experimental:c11atomics
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _CRT_SECURE_NO_WARNINGS
    )
endif()

# The lifted translation units are machine-generated and can run to hundreds of
# thousands of lines each, so they get their own flags:
#
#   - warnings off. They are about generated code and nobody will act on them,
#     and at this volume they bury the diagnostics that matter.
#   - no floating-point contraction. arm64 fuses a multiply and an add by
#     default; a PPC fmadd is spelt as one where the lifter means one.
#   - /bigobj on MSVC. A single lifted file routinely passes the 65,536-section
#     limit of the default object format.
# F181 (perf): PORT_PPU_O2=ON compiles the lifted PPU TUs at -O2 instead of -O1 (main guest thread is
# ~45% busy in the menu, ~20% in the world). Costs roughly 2x compile time/memory for those 8 TUs, so it
# is opt-in: cmake -S . -B <build> -DPORT_PPU_O2=ON.
option(PORT_PPU_O2 "F181: lifted PPU sources at -O2 instead of -O1" OFF)
if(PORT_PPU_O2)
    set(F181_PPU_OPT "-O2")
else()
    set(F181_PPU_OPT "-O1")
endif()
if(MSVC)
    if(PORT_OPT)
        # v170P: -O1 (clang-cl /O1 means -Os, hence /clang:), no FP contraction:
        # clang-cl ignores a bare -ffp-contract=off and AVX2 brings FMA, which
        # would fuse a separately rounded PPC fmul+fadd pair.
        set_source_files_properties(${RECOMP_SOURCES} PROPERTIES
            COMPILE_OPTIONS "/w;/bigobj;/clang:${F181_PPU_OPT};/clang:-ffp-contract=off")
    else()
        set_source_files_properties(${RECOMP_SOURCES} PROPERTIES
            COMPILE_OPTIONS "/w;/bigobj")
    endif()
else()
    if(PORT_OPT)
        set_source_files_properties(${RECOMP_SOURCES} PROPERTIES
            COMPILE_OPTIONS "-w;-ffp-contract=off;${F181_PPU_OPT}")
    else()
        set_source_files_properties(${RECOMP_SOURCES} PROPERTIES
            COMPILE_OPTIONS "-w;-ffp-contract=off")
    endif()
endif()
# QP (2026-10-16): PORT_PPU_O2_ONLY=<name;name> (basenames, e.g. ppu_recomp_001;ppu_recomp_003) compiles only
# these lifted TUs at -O2 (with PORT_PPU_O2=OFF): -O2 where the main thread's PPU time really is, without paying
# the full -O2 compile of all 8 TUs. Empty (default) = unchanged.
set(PORT_PPU_O2_ONLY "" CACHE STRING "QP: lifted PPU TUs (basename) at -O2 when PORT_PPU_O2=OFF")
if(PORT_OPT AND PORT_PPU_O2_ONLY AND NOT PORT_PPU_O2)
    foreach(_qp_src IN LISTS RECOMP_SOURCES)
        get_filename_component(_qp_n "${_qp_src}" NAME_WE)
        list(FIND PORT_PPU_O2_ONLY "${_qp_n}" _qp_i)
        if(_qp_i GREATER -1)
            if(MSVC)
                set_source_files_properties(${_qp_src} PROPERTIES
                    COMPILE_OPTIONS "/w;/bigobj;/clang:-O2;/clang:-ffp-contract=off")
            else()
                set_source_files_properties(${_qp_src} PROPERTIES
                    COMPILE_OPTIONS "-w;-ffp-contract=off;-O2")
            endif()
            message(STATUS "QP: ${_qp_n} at -O2 (PORT_PPU_O2_ONLY)")
        endif()
    endforeach()
endif()

# Inline the ordinary guest-memory fast path in lifted PPU code. The runtime's
# precomputed gates preserve all special-address, diagnostic, bounds and SPU
# reservation behavior by falling back to the existing vm_* helpers.
option(PORT_PPU_INLINE_LOADS "Inline fast guest loads in lifted PPU sources" OFF)
option(PORT_PPU_INLINE_STORES "Also inline coherency-gated guest stores" OFF)
if(PORT_PPU_INLINE_STORES AND NOT PORT_PPU_INLINE_LOADS)
    message(FATAL_ERROR "PORT_PPU_INLINE_STORES requires PORT_PPU_INLINE_LOADS")
endif()
if(PORT_PPU_INLINE_LOADS)
    foreach(_ppu_src IN LISTS RECOMP_SOURCES)
        set_property(SOURCE "${_ppu_src}" APPEND PROPERTY COMPILE_DEFINITIONS PPU_INLINE_LOADS)
        if(NOT PORT_PPU_INLINE_STORES)
            set_property(SOURCE "${_ppu_src}" APPEND PROPERTY COMPILE_DEFINITIONS
                PPU_NO_INLINE_STORES PPU_NO_INLINE_STVX)
        endif()
    endforeach()
    message(STATUS "Lifted PPU inline memory: loads ON, stores ${PORT_PPU_INLINE_STORES}")
else()
    message(STATUS "Lifted PPU inline memory: disabled")
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
