#ifndef I3D_LINE3D_PP_SERIALIZATION_H_
#define I3D_LINE3D_PP_SERIALIZATION_H_

/*
Line3D++ - Line-based Multi View Stereo
Copyright (C) 2015  Manuel Hofer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <boost/filesystem.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/split_free.hpp>

#include <fstream>

/**
 * Line3D++ - Serialization
 * ====================
 * Handles serialization of view- and
 * matching-data to the hard drive
 * using boost.
 * ====================
 * Author: M.Hofer, 2015
 */

namespace L3DPP
{
    // serialization
    template <typename T>
    inline void
    serializeToFile(std::string file, T const& data, bool binary = true)
    {
        std::ofstream os(file.c_str(), binary? std::ios::binary : std::ios::out);
        boost::archive::binary_oarchive ar(os);
        ar & boost::serialization::make_nvp("data", data);
    }

    template <typename T>
    inline void
    serializeFromFile(std::string file, T& data, bool binary = true)
    {
        std::ifstream is(file.c_str(), binary? std::ios::binary : std::ios::in);
        if(is.bad()) {
            std::cout << "[L3D++] serializeFromFileArchive(): File '" << file << "'' could not be opened " << std::endl;
          exit(1);
        }
        boost::archive::binary_iarchive ar(is);
        ar & boost::serialization::make_nvp("data", data);
    }
}

#endif //I3D_LINE3D_PP_SERIALIZATION_H_
