#include <utils/YAMLLoader.h>

namespace vo {
    namespace util {

        YAMLLoader::YAMLLoader(const std::string filename):
            config_(YAML::LoadFile(filename))
        {};

        YAMLLoader::~YAMLLoader(){};

        // Special handler for loading intrinsics matrix from YAML file
        template<> vo::util::Mat3f YAMLLoader::get_value<vo::util::Mat3f>(const std::string key) {
            if (!config_[key]) {
                throw std::invalid_argument("File does not contain key '" + key + "'.");
            }

            vo::util::Mat3f value =  vo::util::Mat3f::Map(config_[key].as<std::vector<float>>().data(), 3, 3);

            return value;
        };

    } // namespace util
} // namespace vo