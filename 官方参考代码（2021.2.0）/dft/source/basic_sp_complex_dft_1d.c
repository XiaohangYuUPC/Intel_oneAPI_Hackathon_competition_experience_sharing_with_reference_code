/*******************************************************************************
* Copyright 2011-2021 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
! Content:
! A simple example of single-precision complex-to-complex in-place 1D
! FFT using Intel(R) Math Kernel Library (Intel(R) MKL) DFTI
!
!****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "mkl_service.h"
#include "mkl_dfti.h"

/* Define the format to printf MKL_LONG values */
#if !defined(MKL_ILP64)
#define LI "%li"
#else
#define LI "%lli"
#endif

static void init(MKL_Complex8 *data, int N, int H);
static int verify(MKL_Complex8 *data, int N, int H);

int main(void)
{
    /* Size of 1D transform */
    int N = 7;

    /* Arbitrary harmonic used to verify FFT */
    int H = -N/2;

    /* Pointer to input/output data */
    MKL_Complex8 *x = NULL;

    /* Execution status */
    MKL_LONG status = 0;

    DFTI_DESCRIPTOR_HANDLE hand = NULL;

    char version[DFTI_VERSION_LENGTH];

    DftiGetValue(0, DFTI_VERSION, version);
    printf("%s\n", version);

    printf("Example basic_sp_complex_dft_1d\n");
    printf("Forward and backward single-precision complex in-place 1D FFTs\n");
    printf("Configuration parameters:\n");
    printf(" DFTI_PRECISION      = DFTI_SINGLE\n");
    printf(" DFTI_FORWARD_DOMAIN = DFTI_COMPLEX\n");
    printf(" DFTI_DIMENSION      = 1\n");
    printf(" DFTI_LENGTHS        = {%d}\n", N);

    printf("Create DFTI descriptor\n");
    status = DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_COMPLEX,
                                  1, (MKL_LONG)N);
    if (status != DFTI_NO_ERROR) goto failed;

    printf("Commit DFTI descriptor\n");
    status = DftiCommitDescriptor(hand);
    if (status != DFTI_NO_ERROR) goto failed;

    printf("Allocate input array\n");
    x = (MKL_Complex8*)mkl_malloc(N*sizeof(MKL_Complex8), 64);
    if (x == NULL) goto failed;

    printf("Initialize input for forward transform\n");
    init(x, N, H);

    printf("Compute forward transform\n");
    status = DftiComputeForward(hand, x);
    if (status != DFTI_NO_ERROR) goto failed;

    printf("Verify the result of forward FFT\n");
    status = verify(x, N, H);
    if (status != 0) goto failed;

    printf("Initialize input for backward transform\n");
    init(x, N, -H);

    printf("Compute backward transform\n");
    status = DftiComputeBackward(hand, x);
    if (status != DFTI_NO_ERROR) goto failed;

    printf("Verify the result of backward FFT\n");
    status = verify(x, N, H);
    if (status != 0) goto failed;

 cleanup:

    printf("Release the DFTI descriptor\n");
    DftiFreeDescriptor(&hand);

    printf("Free data array\n");
    mkl_free(x);

    printf("TEST %s\n", (status == 0) ? "PASSED" : "FAILED");
    return status;

 failed:
    printf(" ERROR, status = "LI"\n", status);
    status = 1;
    goto cleanup;
}

/* Compute (K*L)%M accurately */
static float moda(int K, int L, int M)
{
    return (float)(((long long)K * L) % M);
}

/* Initialize array with harmonic H */
static void init(MKL_Complex8 *x, int N, int H)
{
    float TWOPI = 6.2831853071795864769f, phase;
    int n;

    for (n = 0; n < N; n++) {
        phase  = moda(n, H, N) / N;
        x[n].real = cosf(TWOPI * phase) / N;
        x[n].imag = sinf(TWOPI * phase) / N;
    }
}

/* Verify that x has unit peak at H */
static int verify(MKL_Complex8 *x, int N, int H)
{
    float err, errthr, maxerr;
    int n;

    /*
     * Note, this simple error bound doesn't take into account error of
     * input data
     */
    errthr = 5.0f * logf((float) N) / logf(2.0f) * FLT_EPSILON;
    printf(" Verify the result, errthr = %.3lg\n", errthr);

    maxerr = 0.0f;
    for (n = 0; n < N; n++) {
        float re_exp = 0.0f, im_exp = 0.0f, re_got, im_got;

        if ((n-H)%N==0) re_exp = 1.0f;

        re_got = x[n].real;
        im_got = x[n].imag;
        err  = fabsf(re_got - re_exp) + fabsf(im_got - im_exp);
        if (err > maxerr) maxerr = err;
        if (!(err < errthr)) {
            printf(" x[%i]: ", n);
            printf(" expected (%.7g,%.7g), ", re_exp, im_exp);
            printf(" got (%.7g,%.7g), ", re_got, im_got);
            printf(" err %.3lg\n", err);
            printf(" Verification FAILED\n");
            return 1;
        }
    }
    printf(" Verified, maximum error was %.3lg\n", maxerr);
    return 0;
}
