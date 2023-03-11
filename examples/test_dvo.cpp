#include <boost/program_options.hpp>
#include <iostream>

#include <chrono>
#include <iomanip>

using Time = std::chrono::high_resolution_clock;
using double_sec = std::chrono::duration<double>;
using time_point = std::chrono::time_point<Time, double_sec>;

namespace po = boost::program_options;

#include <core/DenseVisualOdometry.h>


bool parse_arguments(
    int argc, char*argv[], std::string& type, std::string& intrinsics_file, std::string& config_file, int* size
) {
    try {

        po::options_description desc("test_dvo usage");
        desc.add_options()
            ("help,h", "produce help message")
            ("type,t", po::value<std::string>(&type)->required(), "Benchmark type {tum-fr1, test}")
            ("intrinsics-file,i", po::value<std::string>(&intrinsics_file)->required(), "Camera Intrinsics YAML file")
            ("config-file,c", po::value<std::string>(&config_file)->required(), "Visual odometry config YAML file")
            ("size", po::value<int>(size)->default_value(-1), "Number of samples to use")
        ;

        po::positional_options_description pos;
        pos.add("type", -1);

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return false;
        }

        po::notify(vm);

    } catch (std::exception& e) {
        std::cerr << "Error " << e.what() << std::endl;
        return false;

    } catch(...) {
        std::cerr << "Unknown error!" << std::endl;
        return false;
    }

    return true;
}


int main(int argc, char *argv[]){

    /* CLI Argument Parser */
    std::string type, config_file, intrinsics_file;
    int size;
    if(!parse_arguments(argc, argv, type, intrinsics_file, config_file, &size))
        return 1;

    vo::core::DenseVisualOdometry dvo = vo::core::DenseVisualOdometry::load_from_yaml(config_file);
    std::cout << "Created DVO" << std::endl;
    dvo.update_camera_info(intrinsics_file);
    std::cout << "Init camera" << std::endl;

    time_point start = Time::now();
    time_point end = Time::now();

    std::cout << "Elapsed time: " << (end - start).count() << std::endl;

    return 0;
}