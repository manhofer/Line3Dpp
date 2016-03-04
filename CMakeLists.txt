cmake_minimum_required(VERSION 2.6)

set (CMAKE_BUILD_TYPE Release)
SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

##-----------------------------------------------------------------------------
## OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
set (EXTRA_INC_DIRS
    ${EXTRA_INC_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
set (EXTRA_LIBRARIES
    ${EXTRA_LIBRARIES}
    ${OpenCV_LIBS}
)

link_directories(${OPENCV_LIB_DIR})

# check for OpenCV >= 3
if(${OpenCV_VERSION_MAJOR} GREATER "2")
  message("using LSD from OpenCV 3")
  SET(L3DPP_OPENCV3 1)
else(${OpenCV_VERSION_MAJOR} GREATER "2")
  SET(lsd_SOURCES lsd/lsd.cpp lsd/lsd_opencv.cpp lsd/lsd_wrap.cpp)
  SET(lsd_HEADERS lsd/lsd.hpp lsd/lsd_opencv.hpp lsd/lsd_wrap.hpp)
  add_library(line3Dpp_lsd SHARED ${lsd_SOURCES} ${lsd_HEADERS})
  target_link_libraries(line3Dpp_lsd ${OpenCV_LIBS})
endif(${OpenCV_VERSION_MAJOR} GREATER "2")

##-----------------------------------------------------------------------------
## Eigen
FIND_PACKAGE(Eigen REQUIRED)

set (EXTRA_INC_DIRS
    ${EXTRA_INC_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}/unsupported
)
include_directories(${EIGEN3_INCLUDE_DIR})

##-----------------------------------------------------------------------------
## CUDA
  FIND_PACKAGE(CUDA)
  IF(CUDA_FOUND)
    set (EXTRA_INC_DIRS
        ${EXTRA_INC_DIRS}
        ${CUDA_INCLUDE_DIRS}
		${CUDA_SDK_INCLUDE_DIR}
    )
	cuda_include_directories(${EXTRA_INC_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
    set (EXTRA_LIBRARIES
        ${EXTRA_LIBRARIES}
        ${CUDA_LIBS}
        ${CUDA_cublas_LIBRARY}
    )
  ENDIF(CUDA_FOUND)

##-----------------------------------------------------------------------------
## Boost
set(Boost_USE_STATIC_LIBS        OFF)
set(Boost_USE_STATIC_RUNTIME     OFF)
FIND_PACKAGE(Boost REQUIRED COMPONENTS serialization filesystem system thread)

if(Boost_FOUND)
        set (EXTRA_INC_DIRS
                ${EXTRA_INC_DIRS}
                ${Boost_INCLUDE_DIRS}
        )

          set (EXTRA_LIBRARIES
                ${EXTRA_LIBRARIES}
                ${Boost_LIBRARIES}
          )
endif(Boost_FOUND)

##-----------------------------------------------------------------------------
## CERES
find_package(Ceres)
if(Ceres_FOUND)
  include_directories(${CERES_INCLUDES})
  set(EXTRA_LIBRARIES ${EXTRA_LIBRARIES} optimized ceres)
endif(Ceres_FOUND)

if(NOT Ceres_FOUND AND WIN32)
  # use precompiled CERES version
  set(PRECOMPILED_DIR ${CMAKE_SOURCE_DIR}/libs/precompiled)

  # only 64bit CERES is included
  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set(PRECOMPILED_LIB lib64)
    set(PRECOMPILED_LIB_DIR_FILTER bin64/ )
    deep_copy_file(${PRECOMPILED_DIR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE} dll ${PRECOMPILED_LIB_DIR_FILTER} release)

    UNSET(CERES_LIBDIR CACHE)
    FIND_PATH(CERES_LIBDIR NAMES ceres.lib PATHS ${PRECOMPILED_DIR}/ceres/${PRECOMPILED_LIB}/vc10)
    FIND_PATH(CERES_INCLUDE_DIR NAMES ceres/ceres.h PATHS ${PRECOMPILED_DIR}/ceres/include)

    if(CERES_LIBDIR AND CERES_INCLUDE_DIR)
      find_package_handle_standard_args(CERES DEFAULT_MSG CERES_INCLUDE_DIR)
      set(EXTRA_LIBRARIES ${EXTRA_LIBRARIES} optimized ceres)
      include_directories(${CERES_INCLUDE_DIR})
      link_directories(${CERES_LIBDIR})
      set(Ceres_FOUND 1)
    endif(CERES_LIBDIR AND CERES_INCLUDE_DIR)
    
  endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
		   
endif(NOT Ceres_FOUND AND WIN32)

##-----------------------------------------------------------------------------
## OpenMP
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
    message("OPENMP_FOUND")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF(OPENMP_FOUND)

##-----------------------------------------------------------------------------
## RapidJSON
FIND_PACKAGE(RapidJSON)

add_definitions(-frounding-math)

## switches for external (optional) libs
IF(CUDA_FOUND)
    option(APP_LINE_3D++_USE_CUDA "Line3D++: use CUDA" ON)
ELSE(CUDA_FOUND)
    option(APP_LINE_3D++_USE_CUDA "Line3D++: use CUDA" OFF)
ENDIF(CUDA_FOUND)

IF(OPENMP_FOUND)
    option(APP_LINE_3D++_USE_OPENMP "Line3D++: use OpenMP" ON)
ELSE(OPENMP_FOUND)
    option(APP_LINE_3D++_USE_OPENMP "Line3D++: use OpenMP" OFF)
ENDIF(OPENMP_FOUND)

IF(Ceres_FOUND)
    option(APP_LINE_3D++_USE_CERES "Line3D++: use Ceres" ON)
ELSE(Ceres_FOUND)
    option(APP_LINE_3D++_USE_CERES "Line3D++: use Ceres" OFF)
ENDIF(Ceres_FOUND)

if(CUDA_FOUND AND APP_LINE_3D++_USE_CUDA)
    SET(L3DPP_CUDA 1)
endif(CUDA_FOUND AND APP_LINE_3D++_USE_CUDA)

if(OPENMP_FOUND AND APP_LINE_3D++_USE_OPENMP)
    SET(L3DPP_OPENMP 1)
endif(OPENMP_FOUND AND APP_LINE_3D++_USE_OPENMP)

if(Ceres_FOUND AND APP_LINE_3D++_USE_CERES)
    SET(L3DPP_CERES 1)
endif(Ceres_FOUND AND APP_LINE_3D++_USE_CERES)

## create header with defines
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/configLIBS.h.in ${CMAKE_CURRENT_BINARY_DIR}/configLIBS.h)

#---- combine external libs -----
IF(L3DPP_OPENCV3)
  set(ALL_LIBRARIES ${EXTRA_LIBRARIES})
ELSE(L3DPP_OPENCV3)
  set(ALL_LIBRARIES line3Dpp_lsd ${EXTRA_LIBRARIES})
ENDIF(L3DPP_OPENCV3)

#---- Add Line3D++ library----
SET(Line3D_HEADERS line3D.h view.h clustering.h universe.h serialization.h commons.h dataArray.h segment3D.h optimization.h sparsematrix.h cudawrapper.h configLIBS.h)
IF(L3DPP_CUDA)
        SET(Line3D_SOURCES line3D.cc view.cc clustering.cc optimization.cc sparsematrix.cc cudawrapper.cu)
ELSE(L3DPP_CUDA)
        SET(Line3D_SOURCES line3D.cc view.cc optimization.cc sparsematrix.cc clustering.cc)
ENDIF(L3DPP_CUDA)

IF(L3DPP_CUDA)
  CUDA_ADD_LIBRARY(line3Dpp SHARED ${Line3D_SOURCES} ${Line3D_HEADERS})
ELSE(L3DPP_CUDA)
  ADD_LIBRARY(line3Dpp SHARED ${Line3D_SOURCES} ${Line3D_HEADERS})
ENDIF(L3DPP_CUDA)
target_link_libraries(line3Dpp ${ALL_LIBRARIES})

option(APP_LINE_3D++_BUILD_EXECUTABLES "Line3D++: build executables" ON)

IF(APP_LINE_3D++_BUILD_EXECUTABLES)

#----- Add main for VisualSfM --------
add_executable(runLine3Dpp_vsfm main_vsfm.cpp)
target_link_libraries(runLine3Dpp_vsfm line3Dpp)
target_link_libraries(runLine3Dpp_vsfm ${ALL_LIBRARIES})

#----- Add main for bundler --------
add_executable(runLine3Dpp_bundler main_bundler.cpp)
target_link_libraries(runLine3Dpp_bundler line3Dpp)
target_link_libraries(runLine3Dpp_bundler ${ALL_LIBRARIES})

#----- Add main for mavmap --------
add_executable(runLine3Dpp_mavmap main_mavmap.cpp)
target_link_libraries(runLine3Dpp_mavmap line3Dpp)
target_link_libraries(runLine3Dpp_mavmap ${ALL_LIBRARIES})

#----- Add main for Pix4D --------
add_executable(runLine3Dpp_pix4d main_pix4d.cpp)
target_link_libraries(runLine3Dpp_pix4d line3Dpp)
target_link_libraries(runLine3Dpp_pix4d ${ALL_LIBRARIES})

#----- Add main for OpenMVG --------
IF(RapidJSON_FOUND)
add_executable(runLine3Dpp_openmvg main_openmvg.cpp)
target_link_libraries(runLine3Dpp_openmvg line3Dpp)
target_link_libraries(runLine3Dpp_openmvg ${ALL_LIBRARIES})
ENDIF(RapidJSON_FOUND)

ENDIF(APP_LINE_3D++_BUILD_EXECUTABLES)


