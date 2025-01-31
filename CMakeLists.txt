

project(lidarDataParser LANGUAGES CXX)
cmake_minimum_required(VERSION 3.16)


include(CheckCXXCompilerFlag)

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))

   # add_compile_options(
   #     -Werror
   #     -Wall
   #     -Wextra
   #     -Wpedantic
   #     -Wcast-align
   #     -Wcast-qual
   #     -Wconversion
   #     -Wctor-dtor-privacy
   #     -Wenum-compare
   #     -Wfloat-equal
   #     -Wnon-virtual-dtor
  #      -Wold-style-cast
  #      -Woverloaded-virtual
  #      -Wredundant-decls
  #      -Wsign-conversion
  #      -Wsign-promo
  #  )
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    check_cxx_compiler_flag(-std=c++17 SUPPORTS_CXX17)
    
    if(SUPPORTS_CXX17)
        add_compile_options(-std=c++17)
    else()
        message(FATAL_ERROR "Compiler doesn't support C++17")
    endif()
endif()

if(NOT CMAKE_CXX_EXTENSIONS)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

find_package(PCL REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})




include_directories(${PCL_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})



include_directories(${PCAP_INCLUDE_DIRS})

link_directories(${PCAP_LIBRARIES})




link_directories(${libusb_LIBRARIES})

link_directories(${libusb_INCLUDE_DIR})




include_directories(${VTK_INCLUDE_DIRS})

link_directories(${VTK_LIBRARY_DIRS})


find_package(PkgConfig)
pkg_check_modules(PC_FLANN flann)
set(FLANN_DEFINITIONS ${PC_FLANN_CFLAGS_OTHER})

find_path(FLANN_INCLUDE_DIR flann/flann.hpp
    HINTS ${PC_FLANN_INCLUDEDIR} ${PC_FLANN_INCLUDE_DIRS})

find_library(FLANN_LIBRARY flann
    HINTS ${PC_FLANN_LIBDIR} ${PC_FLANN_LIBRARY_DIRS})

set(FLANN_INCLUDE_DIRS ${FLANN_INCLUDE_DIR})
set(FLANN_LIBRARIES ${FLANN_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Flann DEFAULT_MSG
    FLANN_LIBRARY FLANN_INCLUDE_DIR)

mark_as_advanced(FLANN_LIBRARY FLANN_INCLUDE_DIR)


FILE (GLOB source_files src/*.cpp )

add_executable(lidarDataParser ${source_files})


target_link_libraries(lidarDataParser ${PCL_LIBRARIES} ${EIGEN3_INCLUDE_DIRS} ${PCAP_LIBRARIES} ${libusb_LIBRARIES} ${VTK_INCLUDE_DIRS} ${FLANN_INCLUDE_DIR})