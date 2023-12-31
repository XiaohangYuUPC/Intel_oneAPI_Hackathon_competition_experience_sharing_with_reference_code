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
! A simple example of double-precision complex-to-complex in-place 3D
! FFT using Intel(R) Math Kernel Library (Intel(R) MKL) DFTI
!
!****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "mkl_service.h"
#include "mkl_dfti.h"

static void init(MKL_Complex16 *data,
                 int N1, int N2, int N3,
                 int H1, int H2, int H3);
static int verify(MKL_Complex16 *data,
                  int N1, int N2, int N3,
                  int H1, int H2, int H3);

/* Define the format to printf MKL_LONG values */
#if !defined(MKL_ILP64)
#define LI "%li"
#else
#define LI "%lli"
#endif

int main(void)
{
    /* Sizes of 3D transform */
    int N1 = 7, N2 = 13, N3 = 5;
    MKL_LONG N[3] = {N3, N2, N1};

    /* Arbitrary harmonic used to verify FFT */
    int H1 = -2, H2 = -3, H3 = -4;

    /* Pointer to input/output data */
    MKL_Complex16 *data = NULL;

    /* Execution status */
    MKL_LONG status = 0;

    DFTI_DESCRIPTOR_HANDLE hand = NULL;

    char version[DFTI_VERSION_LENGTH];

    DftiGetValue(0, DFTI_VERSION, version);
    printf("%s\n", version);

    printf("Example basic_dp_complex_dft_3d\n");
    printf("Forward and backward double-precision complex 3D transforms\n");
    printf("Configuration parameters:\n");
    printf(" DFTI_PRECISION      = DFTI_DOUBLE\n");
    printf(" DFTI_FORWARD_DOMAIN = DFTI_COMPLEX\n");
    printf(" DFTI_DIMENSION      = 3\n");
    printf(" DFTI_LENGTHS        = {%i, %i, %i}\n", N3, N2, N1);

    printf("Create DFTI descriptor\n");
    status = DftiCreateDescriptor(&hand, DFTI_DOUBLE, DFTI_COMPLEX, 3, N);
    if (status != DFTI_NO_ERROR) goto failed;

    /* Note: It is not necessary to set the input/output strides in this case
          since {0, N2*N1, N1, 1} is the default */
    /*
    MKL_LONG is[4] = {0, N2*N1, N1, 1};
    MKL_LONG os[4] = {0, N2*N1, N1, 1};
    printf("Set input  strides = ");
    printf("{"LI", "LI", "LI", "LI"}\n", is[0], is[1], is[2], is[3]);
    status = DftiSetValue(hand, DFTI_INPUT_STRIDES, is);
    if (status != DFTI_NO_ERROR) goto failed;

    printf("Set output strides = ");
    printf("{"LI", "LI", "LI", "LI"}\n", os[0], os[1], os[2], os[3]);
    status = DftiSetValue(hand, DFTI_OUTPUT_STRIDES, os);
    if (status != DFTI_NO_ERROR) goto failed;
    */

    printf("Commit DFTI descriptor\n");
    status = DftiCommitDescriptor(hand);
    if (status != DFTI_NO_ERROR) goto failed;

    printf("Allocate input array\n");
    data = (MKL_Complex16*)mkl_malloc(N3*N2*N1*sizeof(MKL_Complex16), 64);
    if (data == NULL) goto failed;

    printf("Initialize input for forward transform\n");
    init(data, N1, N2, N3, H1, H2, H3);

    printf("Compute forward transform\n");
    status = DftiComputeForward(hand, data);
    if (status != DFTI_NO_ERROR) goto failed;

    printf("Verify the result of forward FFT\n");
    status = verify(data, N1, N2, N3, H1, H2, H3);
    if (status != 0) goto failed;

    printf("Initialize input for backward transform\n");
    init(data, N1, N2, N3, -H1, -H2, -H3);

    printf("Compute backward transform\n");
    status = DftiComputeBackward(hand, data);
    if (status != DFTI_NO_ERROR) goto failed;

    printf("Verify the result of backward FFT\n");
    status = verify(data, N1, N2, N3, H1, H2, H3);
    if (status != 0) goto failed;

 cleanup:

    printf("Free DFTI descriptor\n");
    DftiFreeDescriptor(&hand);

    printf("Free data array\n");
    mkl_free(data);

    printf("TEST %s\n", (status == 0) ? "PASSED" : "FAILED");
    return status;

 failed:
    printf(" ERROR, status = "LI"\n", status);
    status = 1;
    goto cleanup;
}

/* Compute (K*L)%M accurately */
static double moda(int K, int L, int M)
{
    return (double)(((long long)K * L) % M);
}

/* Initialize array with harmonic {H1, H2, H3} */
static void init(MKL_Complex16 *data,
                 int N1, int N2, int N3,
                 int H1, int H2, int H3)
{
    double TWOPI = 6.2831853071795864769, phase;
    int n1, n2, n3, index;

    /* Generalized strides for row-major addressing of data */
    int S1 = 1, S2 = N1, S3 = N2*N1;

    for (n3 = 0; n3 < N3; n3++) {
        for (n2 = 0; n2 < N2; n2++) {
            for (n1 = 0; n1 < N1; n1++) {
                phase =  moda(n1, H1, N1) / N1;
                phase += moda(n2, H2, N2) / N2;
                phase += moda(n3, H3, N3) / N3;
                index =  n3*S3 + n2*S2 + n1*S1;
                data[index].real = cos(TWOPI * phase) / (N3*N2*N1);
                data[index].imag = sin(TWOPI * phase) / (N3*N2*N1);
            }
        }
    }
}

/* Verify that data(n1,n2,n3) is a peak at H1,H2,H3 */
static int verify(MKL_Complex16 *data,
                  int N1, int N2, int N3,
                  int H1, int H2, int H3)
{
    double err, errthr, maxerr;
    int n1, n2, n3, index;

    /* Generalized strides for row-major addressing of data */
    int S1 = 1, S2 = N1, S3 = N2*N1;

    /*
     * Note, this simple error bound doesn't take into account error of
     * input data
     */
    errthr = 5.0 * log((double) N3*N2*N1) / log(2.0) * DBL_EPSILON;
    printf(" Verify the result, errthr = %.3lg\n", errthr);

    maxerr = 0.0;
    for (n3 = 0; n3 < N3; n3++) {
        for (n2 = 0; n2 < N2; n2++) {
            for (n1 = 0; n1 < N1; n1++) {
                double re_exp = 0.0, im_exp = 0.0, re_got, im_got;

                if ((n1-H1)%N1==0 && (n2-H2)%N2==0 && (n3-H3)%N3==0) {
                    re_exp = 1.0;
                }

                index =  n3*S3 + n2*S2 + n1*S1;
                re_got = data[index].real;
                im_got = data[index].imag;
                err  = fabs(re_got - re_exp) + fabs(im_got - im_exp);
                if (err > maxerr) maxerr = err;
                if (!(err < errthr)) {
                    printf(" data[%i][%i][%i]: ", n3, n2, n1);
                    printf(" expected (%.17lg,%.17lg), ", re_exp, im_exp);
                    printf(" got (%.17lg,%.17lg), ", re_got, im_got);
                    printf(" err %.3lg\n", err);
                    printf(" Verification FAILED\n");
                    return 1;
                }
            }
        }
    }
    printf(" Verified, maximum error was %.3lg\n", maxerr);
    return 0;
}
