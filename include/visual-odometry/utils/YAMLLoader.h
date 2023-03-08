#ifndef VO_UTIL_YAML_LOADER
#define VO_UTIL_YAML_LOADER

#include <string>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include <utils/types.h>


namespace vo {
    namespace util {

        class YAMLLoader {
            public:
                YAMLLoader(const std::string filename);
                ~YAMLLoader();

                template<typename T>
                T get_value(const std::string key);

            private:
                YAML::Node config_;
        };

        template<> vo::util::Mat3f YAMLLoader::get_value<vo::util::Mat3f>(const std::string key);

        template<typename T>
        T YAMLLoader::get_value(const std::string key) {
            if (!config_[key]) {
                throw std::invalid_argument("File does not contain key '" + key + "'.");
            }

            T value = config_[key].as<T>();

            return value;
        };

    } // namespace util
} // namespace vo

#endif
