#
# The script defines the following variables:
#
##############################################################################
#  Note: Removed everything related to CUDA_SDK_ROOT_DIR and only left this as 
#        a possible environment variable to set the SDK directory. 
#        Include file will be: CUDA_CUT_INCLUDE_DIR
#        Cutil library:        CUDA_CUT_LIBRARY
##############################################################################
# 
#
#  CUDA_SDK_ROOT_DIR     -- Path to the CUDA SDK.  Use this to find files in the
#                           SDK.  This script will not directly support finding
#                           specific libraries or headers, as that isn't
#                           supported by NVIDIA.  If you want to change
#                           libraries when the path changes see the
#                           FindCUDA.cmake script for an example of how to clear
#                           these variables.  There are also examples of how to
#                           use the CUDA_SDK_ROOT_DIR to locate headers or
#                           libraries, if you so choose (at your own risk).
#
#  This code is licensed under the MIT License.  See the FindCUDASDK.cmake script
#  for the text of the license.

# The MIT License
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
###############################################################################

# FindCUDASDK.cmake

# # Check to see if the CUDA_SDK_ROOT_DIR has changed,
# # if it has then clear the cache variable, so that it will be detected again.
# if(NOT "${CUDA_SDK_ROOT_DIR}" STREQUAL "${CUDA_SDK_ROOT_DIR_INTERNAL}")
#   # No specific variables to catch.  Use this kind of code before calling
#   # find_package(CUDA) to clean up any variables that may depend on this path.
# 
#   #   unset(MY_SPECIAL_CUDA_SDK_INCLUDE_DIR CACHE)
#   #   unset(MY_SPECIAL_CUDA_SDK_LIBRARY CACHE)
# endif()
# 
# ########################
# # Look for the SDK stuff
# find_path(CUDA_SDK_ROOT_DIR cutil.h
#   PATH_SUFFIXES  "common/inc" "C/common/inc"
#   "$ENV{NVSDKCUDA_ROOT}"
#   "[HKEY_LOCAL_MACHINE\\SOFTWARE\\NVIDIA Corporation\\Installed Products\\NVIDIA SDK 10\\Compute;InstallDir]"
#   "/Developer/GPU\ Computing/C"
#   )
# 
# # fallback method for determining CUDA_SDK_ROOT_DIR in case the previous one failed!
# if (NOT CUDA_SDK_ROOT_DIR)
#   find_path(CUDA_SDK_ROOT_DIR C/common/inc/cutil.h
#   "$ENV{NVSDKCUDA_ROOT}"
#   "[HKEY_LOCAL_MACHINE\\SOFTWARE\\NVIDIA Corporation\\Installed Products\\NVIDIA SDK 10\\Compute;InstallDir]"
#   "/Developer/GPU\ Computing/C"
#   )
# endif()

# Keep the CUDA_SDK_ROOT_DIR first in order to be able to override the
# environment variables.
set(CUDA_SDK_SEARCH_PATH
  "${CUDA_SDK_ROOT_DIR}"
  "${CUDA_TOOLKIT_ROOT_DIR}/local/NVSDK0.2"
  "${CUDA_TOOLKIT_ROOT_DIR}/NVSDK0.2"
  "${CUDA_TOOLKIT_ROOT_DIR}/NV_SDK"
  "${CUDA_TOOLKIT_ROOT_DIR}/NV_CUDA_SDK"
  "$ENV{HOME}/NVIDIA_CUDA_SDK"
  "$ENV{HOME}/NVIDIA_CUDA_SDK_MACOSX"
  "$ENV{HOME}/NVIDIA_GPU_Computing_SDK" 
  "$ENV{NVSDKCUDA_ROOT}"
  "/Developer/CUDA"
  )

# Find include file from the CUDA_SDK_SEARCH_PATH

find_path(CUDA_CUT_INCLUDE_DIR
  cutil.h
  PATHS ${CUDA_SDK_SEARCH_PATH}
  PATH_SUFFIXES "common/inc" "C/common/inc"
  DOC "Location of cutil.h"
  NO_DEFAULT_PATH
  )
# Now search system paths
find_path(CUDA_CUT_INCLUDE_DIR cutil.h DOC "Location of cutil.h")

# mark_as_advanced(CUDA_CUT_INCLUDE_DIR)


# Example of how to find a library in the CUDA_SDK_ROOT_DIR

# cutil library is called cutil64 for 64 bit builds on windows.  We don't want
# to get these confused, so we are setting the name based on the word size of
# the build.

# New library might be called cutil_x86_64 !

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(cuda_cutil_name cutil64)
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(cuda_cutil_name cutil32)
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

find_library(CUDA_CUT_LIBRARY
  NAMES ${cuda_cutil_name} cutil cutil_x86_64 cutil_i386
  PATHS ${CUDA_SDK_SEARCH_PATH}
  # The new version of the sdk shows up in common/lib, but the old one is in lib
  # The very newest installation Path of the SDK is in subdirectory 'C'. Please add this Path to the possible suffixes.
  PATH_SUFFIXES "C/lib" "common/lib" "lib" "C/common/lib" "common/lib"
  DOC "Location of cutil library"
  NO_DEFAULT_PATH
  )
# # Now search system paths
# find_library(CUDA_CUT_LIBRARY NAMES cutil ${cuda_cutil_name} DOC "Location of cutil library")
mark_as_advanced(CUDA_CUT_LIBRARY)
set(CUDA_CUT_LIBRARIES ${CUDA_CUT_LIBRARY})

#############################
# Check for required components
if(CUDA_CUT_INCLUDE_DIR)
  set(CUDASDK_FOUND TRUE)
endif(CUDA_CUT_INCLUDE_DIR)

# set(CUDA_SDK_ROOT_DIR_INTERNAL "${CUDA_SDK_ROOT_DIR}" CACHE INTERNAL
#   "This is the value of the last time CUDA_SDK_ROOT_DIR was set successfully." FORCE)
