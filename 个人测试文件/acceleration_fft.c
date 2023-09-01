#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "fftw3.h"
#include "mkl_dfti.h"
#include <time.h>
#include "mkl.h"
#include <string.h>


#define SIZE1 2048
#define SIZE2 2048
#define EPSILON 100.0
#define NUM_RUNS 1000

typedef struct {
    int status;
    float maxerr;
} VerifyResult;

static void init_r_general(float *x, int N1, int N2, int H1, int H2, int S1, int S2, int type);
static VerifyResult verify_c_fftw3(fftwf_complex *x, int N1, int N2, int H1, int H2);
static VerifyResult verify_c_MKL(MKL_Complex8 *data, int N1, int N2, int H1, int H2);
void init_data(float *data, int size) ;
int compare_data(fftwf_complex *data1, MKL_Complex8 *data2, int size, int* error_count);

int main() {
	/* Set number of threads for MKL */
	/* You can set the number of threads in MKL by yourself */
	//mkl_set_num_threads(8);
	
	/* 2D array dimensions */
    int N1 = SIZE1;
    int N2 = SIZE2;
    int H1 = 1;
    int H2 = N2 / 2;

    MKL_LONG N[] = {N1, N2};
    
    /* Allocate memory and create FFTW plan for 2D real-to-complex FFT */
	float *x_fftw3 = (float *)fftwf_malloc(sizeof(float) * 2 * N1 * (N2 / 2 + 1));
	fftwf_plan r2c_fftw3 = fftwf_plan_dft_r2c_2d(N1, N2, x_fftw3, (fftwf_complex *)x_fftw3, FFTW_ESTIMATE);
    init_r_general(x_fftw3, N1, N2, H1, H2, (N2/2+1)*2, 1, 1);
    
    /* Set MKL stride for 2D real-to-complex FFT */
    MKL_LONG rs[3] = { 0, N1, 1 };
    MKL_LONG cs[3] = { 0, N1 / 2 + 1, 1 };
    DFTI_DESCRIPTOR_HANDLE hand_MKL = NULL;
    
    /* Create and set up MKL descriptor for 2D real-to-complex FFT */
    int status_MKL = DftiCreateDescriptor(&hand_MKL, DFTI_SINGLE, DFTI_REAL, 2, N);
	status_MKL = DftiSetValue(hand_MKL, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	status_MKL = DftiSetValue(hand_MKL, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	status_MKL = DftiSetValue(hand_MKL, DFTI_INPUT_STRIDES, rs);
	status_MKL = DftiSetValue(hand_MKL, DFTI_OUTPUT_STRIDES, cs);
	status_MKL = DftiCommitDescriptor(hand_MKL);
	
	/* Allocate memory for MKL real and complex arrays */
    float *data_real_MKL = (float *)mkl_malloc(N2 * N1 * sizeof(float), 64);
    MKL_Complex8 *data_complex_MKL = (MKL_Complex8 *)mkl_malloc(sizeof(MKL_Complex8) * N1 * (N2 / 2 + 1), 64);
    init_r_general(data_real_MKL, N1, N2, H1, H2, 1, N1, 0);

    printf("\n-------------- Comparative accuracy calculation of 2DFFT for r2c --------------\n");
    printf("\n2D array dimensions: %d x %d\n", N1, N2);
    /* Run FFT computations */
    fftwf_execute(r2c_fftw3);
    printf("FFTW3 computation done.\n");
    status_MKL = DftiComputeForward(hand_MKL, data_real_MKL, data_complex_MKL);
    printf("MKL computation done.\n");
    
    /* Verify results */
    printf("\n---------------------------- Verification Results -----------------------------\n");
    printf("\nVerify FFTW3 results.\n");
    VerifyResult result_fftw3 = verify_c_fftw3((fftwf_complex *)x_fftw3, N1, N2, H1, H2);
    printf(" FFTW3 verification %s\n", result_fftw3.status == 0 ? "PASSED" : "FAILED");
    printf("Verify MKL results.\n");
    VerifyResult result_mkl = verify_c_MKL(data_complex_MKL, N1, N2, H1, H2);
    printf(" MKL verification %s\n", result_mkl.status == 0 ? "PASSED" : "FAILED");
    printf("\n");
    if (result_fftw3.maxerr < result_mkl.maxerr) {
        printf("FFTW3 has higher accuracy (lower error).\n");
    } else if (result_mkl.maxerr < result_fftw3.maxerr) {
        printf("MKL has higher accuracy (lower error).\n");
    } else {
        printf("Both FFTW3 and MKL have the same accuracy.\n");
    }

    /* Clean up */
    fftwf_destroy_plan(r2c_fftw3);
    fftwf_free(x_fftw3);
    mkl_free(data_real_MKL);
    mkl_free(data_complex_MKL);
    DftiFreeDescriptor(&hand_MKL);
    
    /* Multi-run FFTW and MKL for execution time comparison */
    printf("\n------ Multiple runs of FFTW and MKL with random single-precision input ------\n\n");
    printf("2D array dimensions: %d x %d, Total runs: %d\n", N1, N2, NUM_RUNS);
    /* Initialize data */
    int run; 
    float *data_real = (float *)fftwf_malloc(sizeof(float) * N1 * N2 );
    fftwf_complex *data_complex_fftw = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * N1 * N2 );
    data_real_MKL = (float *)mkl_malloc(N2 * N1 * sizeof(float), 64);
    MKL_Complex8 *data_complex_mkl = (MKL_Complex8 *)mkl_malloc(sizeof(MKL_Complex8) * N1 * (N2 / 2 + 1), 64);
    
    /* FFTW FFT */
    double total_time_fftw = 0;
    double total_time_fftw_convert = 0;
    double total_time_mkl = 0;
    double total_time_mkl_convert = 0;
    float total_error = 0; 
    int error_count = 0.0;
    double total_error_count = 0.0;
    //printf("\n");
    for (run = 0; run < NUM_RUNS; run++) {
		printf("\rProgress: [%-50s] %d%%", "**************************************************" + 50 - (run+1)*50/NUM_RUNS, (run+1)*100/NUM_RUNS);
		fflush(stdout);
		
		/* Initialize data */
		init_data(data_real_MKL, N1 * N2);
		
		/* Run FFTW and time it */
        clock_t start_fftw = clock();
        fftwf_plan plan_fftw = fftwf_plan_dft_r2c_2d(N1, N2, data_real_MKL, data_complex_fftw, FFTW_ESTIMATE);
        clock_t start_fftw_convert = clock();
        fftwf_execute(plan_fftw);
        clock_t end_fftw_convert = clock();
        fftwf_destroy_plan(plan_fftw);
        clock_t end_fftw = clock();
        total_time_fftw += (double)(end_fftw - start_fftw) / CLOCKS_PER_SEC;
        total_time_fftw_convert += (double)(end_fftw_convert - start_fftw_convert) / CLOCKS_PER_SEC;
        
        /* Run MKL and time it */
        clock_t start_mkl = clock();
        DFTI_DESCRIPTOR_HANDLE hand_mkl;
        DftiCreateDescriptor(&hand_mkl, DFTI_SINGLE, DFTI_REAL, 2, N);
        //DftiSetValue(hand_mkl, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiCommitDescriptor(hand_mkl);
        clock_t start_mkl_convert = clock();
        DftiComputeForward(hand_mkl, data_real_MKL, data_complex_mkl);
        clock_t end_mkl_convert = clock();
        DftiFreeDescriptor(&hand_mkl);
        clock_t end_mkl = clock();
        total_time_mkl += (double)(end_mkl - start_mkl) / CLOCKS_PER_SEC;
        total_time_mkl_convert += (double)(end_mkl_convert - start_mkl_convert) / CLOCKS_PER_SEC;
        
        /* Compare data and count errors */
        float avg_error = compare_data(data_complex_fftw, data_complex_mkl, N1 * (N2 / 2 + 1), &error_count);
        total_error += avg_error;
        total_error_count += error_count;
    }
    
    /* Calculate average execution time */
    double avg_time_fftw = total_time_fftw / NUM_RUNS;
    double avg_time_fftw_convert = total_time_fftw_convert / NUM_RUNS;
    double avg_time_mkl = total_time_mkl / NUM_RUNS;
    double avg_time_mkl_convert = total_time_mkl_convert / NUM_RUNS;
    /* Calculate average error */
    float final_avg_error = total_error / NUM_RUNS;
    /* Calculate error rate */
    double total_count = (double)NUM_RUNS * 2 * N1 * (N2 / 2 + 1);
	
	/* Print average execution times and error rate */
	printf("Average error: %.3g\n", final_avg_error);
    printf("The percentage of errors exceeding %.3g: %.4f%%, detail numbers: %.0f / %.0f (%d * %d *%d)\n", EPSILON, 100*total_error_count/total_count, (double)total_error_count, total_count, NUM_RUNS, N1, N2);
    printf("\nAverage execution time for the complete FFTW3: %.9f seconds\n", avg_time_fftw);
    printf("Average execution time of FFTW3 operations: %.9f seconds\n", avg_time_fftw_convert);
    printf("Average execution time for the complete MKL: %.9f seconds\n", avg_time_mkl);
    printf("Average execution time of MKL operations: %.9f seconds\n", avg_time_mkl_convert);
	/* Print which method is faster and by how much */
	if (avg_time_fftw < avg_time_mkl) {
		printf("\nOn average, FFTW3 is faster than MKL for the complete process by a factor of %.3f.\n", avg_time_mkl / avg_time_fftw);
	} else {
		printf("\nOn average, MKL is faster than FFTW3 for the complete process by a factor of %.3f.\n", avg_time_fftw / avg_time_mkl);
	}

	if (avg_time_fftw_convert < avg_time_mkl_convert) {
		printf("On average, FFTW3 is faster than MKL for the transformation operation by a factor of %.3f.\n", avg_time_mkl_convert / avg_time_fftw_convert);
	} else {
		printf("On average, MKL is faster than FFTW3 for the transformation operation by a factor of %.3f.\n", avg_time_fftw_convert / avg_time_mkl_convert);
	}

    printf("\n---------------------------- All runs completed -----------------------------\n");

	
    /* Free memory */
    fftwf_free(data_real);
    fftwf_free(data_complex_fftw);
    mkl_free(data_real_MKL);
    mkl_free(data_complex_mkl);
    
    return 0;
}

/* Initialize the data with random numbers */
void init_data(float *data, int size) {
	/* Seed the random number generator with current time */
	srand(time(NULL));
    int i;
    /* Fill the data with random numbers between 0 and 1 */
    for (i = 0; i < size; i++) {
        data[i] = rand() / (float)RAND_MAX;
    }
}

/* Compare two data sets and calculate the error */
int compare_data(fftwf_complex *data1, MKL_Complex8 *data2, int size, int* error_count) {
    /* Initialize error variables */
    float err, maxerr = 0.0f, totalerr = 0.0f;
    *error_count = 0;
    int i;
	
	/* Iterate over each data point */
	for (i = 0; i < size; i++) {
		/* Calculate the difference in real and imaginary parts */
		float re_diff = data1[i][0] - data2[i].real;
		float im_diff = data1[i][1] - data2[i].imag;
		/* Calculate the magnitude of the difference vector */
		err = sqrt(re_diff * re_diff + im_diff * im_diff);
		/* Add the error to the total error */
		totalerr += err;
		/* Update the maximum error if this error is greater */
		if (err > maxerr) maxerr = err;
		/* Increment the error count if error is above the threshold */
		if (err > EPSILON) {
			(*error_count)++;
		}
	}

	/* Return the average error */
    return totalerr / size;
}

/* Compute (K*L)%M accurately */
static float moda(int K, int L, int M)
{
    return (float)(((long long)K * L) % M);
}

/* Initialize data for FFT computation */
static void init_r_general(float *x, int N1, int N2, int H1, int H2, int S1, int S2, int type)
{
    float TWOPI = 6.2831853071795864769f, phase, factor;
    int n1, n2, index;
	
	/* Calculate factor based on type and conditions */
    factor = type ? ((N1-H1%N1)==0 && (N2-H2%N2)==0) ? 1.0f : 2.0f :
                     (2*(N1-H1)%N1==0 && 2*(N2-H2)%N2==0) ? 1.0f : 2.0f;
	
	/* Fill the data array */
    for (n1 = 0; n1 < N1; n1++)
    {
        for (n2 = 0; n2 < N2; n2++)
        {
            /* Calculate phase */
            phase  = moda(n1,H1,N1) / N1;
            phase += moda(n2,H2,N2) / N2;
            index = n1*S1 + n2*S2;
            /* Calculate data value based on phase and factor */
            x[index] = factor * cosf( TWOPI * phase ) / (N1*N2);
        }
    }
}


/* Verify that x has unit peak at H */
static VerifyResult verify_c_fftw3(fftwf_complex *x, int N1, int N2, int H1, int H2)
{
    /* Initialize error variables */
    float err, errthr, maxerr;
    int n1, n2, S1, S2, index;
    VerifyResult result;    

    /* Generalized strides for row-major addressing of x */
    S2 = 1;
    S1 = N2/2+1;

	/* Calculate error threshold */
    errthr = 2.5f * logf( (float)N1*N2 ) / logf(2.0f) * FLT_EPSILON;
    printf(" Check if err is below errthr %.3g\n", errthr);

    maxerr = 0;
    /* Iterate over each data point */
    for (n1 = 0; n1 < N1; n1++)
    {
        for (n2 = 0; n2 < N2/2+1; n2++)
        {
            float re_exp = 0.0, im_exp = 0.0, re_got, im_got;

            /* Check if this point is at peak location */
            if ((( n1-H1)%N1==0 && ( n2-H2)%N2==0) ||
                ((-n1-H1)%N1==0 && (-n2-H2)%N2==0))
            {
                /* The peak should have real part equal to 1 */
                re_exp = 1;
            }

            /* Get the real and imaginary parts of the data */
            index = n1*S1 + n2*S2;
            re_got = x[index][0];
            im_got = x[index][1];
            /* Calculate the error */
            err  = fabsf(re_got - re_exp) + fabsf(im_got - im_exp);
            /* Update the maximum error if this error is greater */
            if (err > maxerr) maxerr = err;
            /* If the error is greater than the threshold, print error details and return failure */
            if (!(err < errthr))
            {
                printf(" x[%i][%i]: ",n1,n2);
                printf(" expected (%.7g,%.7g), ",re_exp,im_exp);
                printf(" got (%.7g,%.7g), ",re_got,im_got);
                printf(" err %.3g\n", err);
                printf(" Verification FAILED\n");            
                result.status = 1;
                result.maxerr = maxerr;
                return result;
            }
        }
    }
    /* If no error was found above the threshold, print success message and return success */
    printf(" Verified,  maximum error was %.3g\n", maxerr);
    result.status = 0;
    result.maxerr = maxerr;
    return result;
}


/* Verify that data has unit peak at H */
static VerifyResult verify_c_MKL(MKL_Complex8 *data, int N1, int N2, int H1, int H2)
{
    /* Initialize error variables */
    float err, errthr, maxerr;
    int n1, n2, index;
    VerifyResult result;

    /* Generalized strides for row-major addressing of data */
    int S1 = 1, S2 = N1/2+1;

	/* Calculate error threshold */
    errthr = 2.5f * logf((float) N2*N1) / logf(2.0f) * FLT_EPSILON;
    printf(" Check if err is below errthr %.3lg\n", errthr);

    maxerr = 0.0f;
    /* Iterate over each data point */
    for (n2 = 0; n2 < N2; n2++) {
        for (n1 = 0; n1 < N1/2+1; n1++) {
            float re_exp = 0.0f, im_exp = 0.0f, re_got, im_got;
			/* Check if this point is at peak location */
            if ((( n1-H1)%N1==0 && ( n2-H2)%N2==0) ||
                ((-n1-H1)%N1==0 && (-n2-H2)%N2==0)
            ) {
                /* The peak should have real part equal to 1 */
                re_exp = 1.0f;
            }

            /* Get the real and imaginary parts of the data */
            index = n2*S2 + n1*S1;
            re_got = data[index].real;
            im_got = data[index].imag;
            /* Calculate the error */
            err  = fabsf(re_got - re_exp) + fabsf(im_got - im_exp);
            /* Update the maximum error if this error is greater */
            if (err > maxerr) maxerr = err;
            /* If the error is greater than the threshold, print error details and return failure */
            if (!(err < errthr)) {
                printf(" data[%i][%i]: ", n2, n1);
                printf(" expected (%.7g,%.7g), ", re_exp, im_exp);
                printf(" got (%.7g,%.7g), ", re_got, im_got);
                printf(" err %.3lg\n", err);
                printf(" Verification FAILED\n");
                result.status = 1;
                result.maxerr = maxerr;
                return result;
            }
        }
    }
    /* If no error was found above the threshold, print success message and return success */
    printf(" Verified,  maximum error was %.3lg\n", maxerr);
    result.status = 0;
    result.maxerr = maxerr;
    return result;
}
