import os
import re
import subprocess
import sys
from pathlib import Path
import platform

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < LooseVersion('3.5.0'):
            raise RuntimeError("CMake >= 3.5.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        build_type = os.environ.get("BUILD_TYPE", "Release")
        build_args = ['--config', build_type]

        # Pile all .so in one place and use $ORIGIN as RPATH
        cmake_args += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE"]
        cmake_args += ["-DCMAKE_INSTALL_RPATH={}".format("$ORIGIN")]

        # Enable python bindings
        cmake_args += ["-DBUILD_PYTHON=ON"]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(build_type.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + build_type]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake',
                               '--build', '.',
                               '--target', os.path.basename(ext.name)
                               ] + build_args,
                              cwd=self.build_temp)


def read(filename: str, encoding: str = "utf8"):
    curr_dir = Path(__file__).resolve().parent
    file = curr_dir / filename
    
    if not file.exists():
        raise FileNotFoundError("Could not find file at '{}'".format(str(file)))

    with file.open(encoding=encoding) as fp:
        content = fp.read()

    return content


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="visual-odometry",
    version=read("VERSION"),
    author="Pedro Fontana",
    author_email="pedro.fontana.1996@gmail.com",
    description="Python lib to perform Visual Odometry",
    long_description=read("README.md"),
    ext_modules=[CMakeExtension("pyvo")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development :: Libraries"
    ],
    install_requires=read("requirements.txt").splitlines(),
    python_requires=">=3.7",
)