#include <iostream>
#include <chrono>
#include <cmath>
#include <complex>
#include <algorithm>
#include <functional>
#include <gsl/gsl-lite.hpp>

#include "FFTW1D.h"

namespace gsl
{
    template <typename To, typename From>
    span<To> reinterpret_span(span<From> s) noexcept
    {
        static_assert(sizeof(To) == sizeof(From), "Size of To and From types must be equal");
        static_assert(std::is_trivially_copyable_v<From>, "From type must be trivially copyable");
        static_assert(std::is_trivially_copyable_v<To>, "To type must be trivially copyable");
        return span<To>(reinterpret_cast<To *>(s.data()), s.size_bytes() / sizeof(To));
    }
}

template <typename T, typename UnaryFoo>
void for_each(T &&v, UnaryFoo f)
{
    std::for_each(v.begin(), v.end(), f);
}

namespace std
{
    double abs(const fftw_complex c) { return std::abs(std::complex(c[0], c[1])); }
}

template <typename T>
auto find_max(const T &arr)
{
    auto m = std::abs(arr[0]);
    for_each(arr, [&](const auto &v)
             { m = std::max(m, std::abs(v)); });
    return m;
}

template <typename T>
size_t find_max_index(const T &arr)
{
    size_t mi{0};
    size_t i{0};
    auto m = std::abs(arr[0]);
    for_each(arr, [&](const auto &v)
             {
        auto av = std::abs(v);
        if(m < av) {
            m = av;
            mi = i;
        }
        i++; });
    return mi;
}

template <typename T>
void normalize(T &arr)
{
    auto max = find_max(arr);
    for_each(arr, [=](decltype(arr[0]) &v)
             { v /= max; });
}

template <typename T>
void squelch(T &&arr, const double level)
{
    for_each(arr, [=](auto &v)
             {
        if(std::abs(v) < level) { 
            typename std::remove_reference<decltype(v)>::type zero{0}; 
            v = zero; 
        } });
}

int test_c2c()
{

    // FFTW1D<> fft(1<<5, FFTW_FORWARD, FFTW_EXHAUSTIVE);
    FFTW1D<std::complex<double>, std::complex<double>> fft(1 << 5, FFTW_FORWARD, FFTW_MEASURE);

    // Create some example data
    double ph = 0;
    const double ph_inc = 2 * M_PI / fft.N;
    for_each(fft.in(), [&ph, ph_inc](auto &sample) {
        std::complex<double> c{1}; // DC
        //c += std::complex<double>(std::sin(-ph), std::cos(-ph));
        c += std::complex<double>(std::sin(ph), std::cos(ph));
        c += 0.5*std::complex<double>(std::sin(2*ph), std::cos(2*ph));
        c += 0.25*std::complex<double>(std::sin(3*ph), std::cos(3*ph));
        c += 0.125*std::complex<double>(std::sin(16*ph), std::cos(16*ph));
        sample = c;
        ph += ph_inc; 
    });
    // for_each(fft.in(), [](auto v) { std::cout << v[0] << " ";});
    // std::cout << "\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    fft.execute();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();

    // Print the results
    std::cout << "FFT " << fft.N << " took " << elapsed_time << " seconds to execute." << std::endl;

    auto out = fft.out();
    normalize(out);
    for_each(out, [](auto v) { std::cout << v << " "; });
    auto max = find_max(out);
    auto max_index = find_max_index(out);
    std::cout << "\n";
    std::cout << max << " @ " << max_index << "\n";
    squelch(out, max * 0.1);
    for_each(out, [](auto v) { std::cout << v << " "; });
    std::cout << "\n";
    max = find_max(out);
    max_index = find_max_index(out);
    std::cout << max << " @ " << max_index << "\n";

    // Clean up

    return 0;
}

int test_r2c()
{

    FFTW1D<double, std::complex<double>> fft(1 << 5, FFTW_FORWARD, FFTW_MEASURE);

    // Create some example data
    double ph = 0;
    const double ph_inc = 2 * M_PI / fft.N;
    for_each(fft.in(), [&ph, ph_inc](auto &sample) {
        sample = 1+sin(ph)+sin(15*ph);
        ph += ph_inc; 
    });

    auto start_time = std::chrono::high_resolution_clock::now();
    fft.execute();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();

    // Print the results
    std::cout << "FFT " << fft.N << " took " << elapsed_time << " seconds to execute." << std::endl;

    auto out = fft.out();
    normalize(out);
    for_each(out, [](auto v) { std::cout << v << " "; });
    auto max = find_max(out);
    auto max_index = find_max_index(out);
    std::cout << "\n";
    std::cout << max << " @ " << max_index << "\n";
    squelch(out, max * 0.1);
    for_each(out, [](auto v) { std::cout << v << " "; });
    std::cout << "\n";
    max = find_max(out);
    max_index = find_max_index(out);
    std::cout << max << " @ " << max_index << "\n";

    // Clean up

    return 0;
}

int main()
{
    test_c2c();
    test_r2c();
    return 0;
}