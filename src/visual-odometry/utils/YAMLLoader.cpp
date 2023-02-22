#include <utils/YAMLLoader.h>

namespace vo {
    namespace util {

        YAMLLoader::YAMLLoader(const std::string filename):
            config_(YAML::LoadFile(filename))
        {}

        YAMLLoader::~YAMLLoader(){};

    } // namespace util
} // namespace vo