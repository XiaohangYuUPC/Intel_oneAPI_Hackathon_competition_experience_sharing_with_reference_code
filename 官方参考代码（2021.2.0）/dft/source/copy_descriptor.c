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
! An example of using DftiCopyDescriptor function.
!
!****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "mkl_service.h"
#include "mkl_dfti.h"

static void init(MKL_Complex16 *x, MKL_LONG N, MKL_LONG H);
static int verify(MKL_Complex16 *x, MKL_LONG N, MKL_LONG H);

/* Define the format to printf MKL_LONG values */
#if !defined(MKL_ILP64)
#define LI "%li"
#else
#define LI "%lli"
#endif

int main(void)
{
    /* Size of 1D transform */
    MKL_LONG N = 1024;

    /* Arbitrary harmonic used to verify FFT */
    MKL_LONG H = -N/2;

    /* Pointer to input/output data */
    MKL_Complex16 *x = NULL;

    /* Execution status */
    MKL_LONG status = 0;

    DFTI_DESCRIPTOR_HANDLE h1 = NULL;
    DFTI_DESCRIPTOR_HANDLE h2 = NULL;

    char version[DFTI_VERSION_LENGTH];

    DftiGetValue(0, DFTI_VERSION, version);
    printf("%s\n", version);

    printf("Example copy_descriptor\n");
    printf("Forward FFT by using a copy of a descriptor\n");
    printf("Configuration parameters:\n");
    printf(" DFTI_PRECISION      = DFTI_DOUBLE\n");
    printf(" DFTI_FORWARD_DOMAIN = DFTI_COMPLEX\n");
    printf(" DFTI_DIMENSION      = 1\n");
    printf(" DFTI_LENGTHS        = "LI"\n", N);

    printf("Create DFTI descriptor h1\n");
    status = DftiCreateDescriptor(&h1, DFTI_DOUBLE, DFTI_COMPLEX, 1, N);
    if (status != DFTI_NO_ERROR) goto failed;

    printf("Commit DFTI descriptor h1\n");
    status = DftiCommitDescriptor(h1);
    if (status != DFTI_NO_ERROR) goto failed;

    printf("Copy descriptor h1 to h2\n");
    status = DftiCopyDescriptor(h1, &h2);
    if (status != DFTI_NO_ERROR) goto failed;

    printf("Free descriptor h1\n");
    DftiFreeDescriptor(&h1);

    printf("Allocate input/output array\n");
    x = (MKL_Complex16*)mkl_malloc(N*sizeof(MKL_Complex16), 64);
    if (x == NULL) goto failed;

    printf("Initialize input for forward transform\n");
    init(x, N, H);

    printf("Compute forward transform using descriptor h2\n");
    status = DftiComputeForward(h2, x);
    if (status != DFTI_NO_ERROR) goto failed;

    printf("Verify the result of forward FFT\n");
    status = verify(x, N, H);
    if (status != 0) goto failed;

 cleanup:

    printf("Release the DFTI descriptors\n");
    DftiFreeDescriptor(&h1); /* ok, h1 should be NULL */
    DftiFreeDescriptor(&h2);

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
static double moda(MKL_LONG K, MKL_LONG L, MKL_LONG M)
{
    return (double)(((long long)K * L) % M);
}

/* Initialize array with harmonic H */
static void init(MKL_Complex16 *x, MKL_LONG N, MKL_LONG H)
{
    double TWOPI = 6.2831853071795864769, phase;
    MKL_LONG n;

    for (n = 0; n < N; n++) {
        phase  = moda(n, H, N) / N;
        x[n].real = cos(TWOPI * phase) / N;
        x[n].imag = sin(TWOPI * phase) / N;
    }
}

/* Verify that x has unit peak at H */
static int verify(MKL_Complex16 *x, MKL_LONG N, MKL_LONG H)
{
    double err, errthr, maxerr;
    MKL_LONG n;

    /*
     * Note, this simple error bound doesn't take into account error of
     * input data
     */
    errthr = 5.0 * log((double) N) / log(2.0) * DBL_EPSILON;
    printf(" Verify the result, errthr = %.3lg\n", errthr);

    maxerr = 0.0;
    for (n = 0; n < N; n++) {
        double re_exp = 0.0, im_exp = 0.0, re_got, im_got;

        if ((n-H)%N==0) re_exp = 1.0;

        re_got = x[n].real;
        im_got = x[n].imag;
        err  = fabs(re_got - re_exp) + fabs(im_got - im_exp);
        if (err > maxerr) maxerr = err;
        if (!(err < errthr)) {
            printf(" x["LI"]: ", n);
            printf(" expected (%.17lg,%.17lg), ", re_exp, im_exp);
            printf(" got (%.17lg,%.17lg), ", re_got, im_got);
            printf(" err %.3lg\n", err);
            printf(" Verification FAILED\n");
            return 1;
        }
    }
    printf(" Verified, maximum error was %.3lg\n", maxerr);
    return 0;
}
