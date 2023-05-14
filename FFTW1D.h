#pragma once

#include <gsl/gsl-lite.hpp>
#include <fftw3.h>


template<typename InT_ = fftw_complex, typename OutT_ = fftw_complex>
class FFTW1D
{
public:

    const int N;
    using InT = InT_;
    using OutT = OutT_;

    FFTW1D(int N_, int sign = FFTW_FORWARD, int flags = FFTW_ESTIMATE) : 
        N(N_),
        in_((InT*)fftw_malloc(sizeof(InT) * N)),
        out_((OutT*)fftw_malloc(sizeof(OutT) * N)),
        plan_(fftw_plan_dft_1d(N, in_, out_, sign, flags))
    {}

    ~FFTW1D()
    {
        fftw_destroy_plan(plan_);
        fftw_free(out_);
        fftw_free(in_);
    }

    auto in() const { return gsl::span<InT>(in_, N); }
    auto out() const { return gsl::span<OutT>(out_, N); }

    void execute()
    {
        fftw_execute(plan_);
    }

private:
    fftw_complex *in_{};
    fftw_complex *out_{};
    fftw_plan plan_{};

};
