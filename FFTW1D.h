#pragma once

#include <gsl/gsl-lite.hpp>
#include <fftw3.h>
#include <complex>

template <typename InT_ = fftw_complex, typename OutT_ = fftw_complex>
class FFTW1D
{
public:
    const int N;
    using InT = InT_;
    using OutT = OutT_;

    FFTW1D(int N_, int sign = FFTW_FORWARD, int flags = FFTW_ESTIMATE) : N(N_)
    {
        initialize(sign, flags);
    }

    ~FFTW1D()
    {
        fftw_destroy_plan(plan_);
        fftw_free(out_);
        fftw_free(in_);
    }

    auto in() const { return gsl::span<InT>(in_, in_size_); }
    auto out() const { return gsl::span<OutT>(out_, out_size_); }

    void execute()
    {
        fftw_execute(plan_);
    }

private:
    InT_ *in_{};
    size_t in_size_{};
    OutT_ *out_{};
    size_t out_size_{};
    fftw_plan plan_{};

    void allocate()
    {
        in_ = (InT *)fftw_malloc(sizeof(InT) * in_size_);
        out_ = (OutT *)fftw_malloc(sizeof(OutT) * out_size_);
    }

    void initialize(int sign, int flags);

};

template <>
void FFTW1D<double, fftw_complex>::initialize(int sign, int flags)
{
    in_size_ = N;
    out_size_ = N / 2 + 1;
    allocate();
    plan_ = fftw_plan_dft_r2c_1d(N, in_, out_, flags);
}

template <>
void FFTW1D<double, std::complex<double>>::initialize(int sign, int flags)
{
    in_size_ = N;
    out_size_ = N / 2 + 1;
    allocate();
    plan_ = fftw_plan_dft_r2c_1d(N, in_, reinterpret_cast<fftw_complex*>(out_), flags);
}

template <>
void FFTW1D<fftw_complex, double>::initialize(int sign, int flags)
{
    in_size_ = N;
    out_size_ = N / 2 + 1;
    allocate();
    plan_ = fftw_plan_dft_c2r_1d(N, in_, out_, flags);
}

template <>
void FFTW1D<std::complex<double>, double>::initialize(int sign, int flags)
{
    in_size_ = N;
    out_size_ = N / 2 + 1;
    allocate();
    plan_ = fftw_plan_dft_c2r_1d(N, reinterpret_cast<fftw_complex*>(in_), out_, flags);
}

template <>
void FFTW1D<fftw_complex, fftw_complex>::initialize(int sign, int flags)
{
    in_size_ = out_size_ = N;
    allocate();
    plan_ = fftw_plan_dft_1d(N, in_, out_, sign, flags);
}

template <>
void FFTW1D<std::complex<double>, std::complex<double>>::initialize(int sign, int flags)
{
    in_size_ = out_size_ = N;
    allocate();
    plan_ = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(in_), reinterpret_cast<fftw_complex*>(out_), sign, flags);
}
