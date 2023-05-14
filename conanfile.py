from conans import ConanFile, CMake

class MyProject(ConanFile):
    name = "myproject"
    version = "1.0"
    generators = "cmake"
    requires = "fftw/3.3.10", "gsl-lite/0.40.0"

    default_options = {"fftw:shared": False}

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
