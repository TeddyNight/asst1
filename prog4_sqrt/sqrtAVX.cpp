#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>


void sqrtAVX(int N,
                float initialGuess,
                float values[],
                float output[])
{

    __m256 kThreshold = _mm256_set1_ps(0.00001f);

    __m256 ones = _mm256_set1_ps(1.f);
    __m256 threes = _mm256_set1_ps(3.f);
    __m256 halfs = _mm256_set1_ps(.5f);
    __m256 abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    for (int i = 0; i < N; i += 8) {
	__m256 vals = _mm256_loadu_ps(values + i);
	__m256 guess = _mm256_set1_ps(initialGuess);
	__m256 g2 = _mm256_mul_ps(guess, guess);

        //float error = fabs(guess * guess * x - 1.f);

	__m256 error1 = _mm256_fmsub_ps(g2, vals, ones);
	__m256 error = _mm256_andnot_ps(abs, error1);

	__m256 mask = _mm256_cmp_ps(error, kThreshold, 30);

	while (_mm256_movemask_ps(mask) != 0) {
		// guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
		// 0.5f * guess * (3.f - x * g2)
		__m256 tmp1 = _mm256_fnmadd_ps(vals, g2, threes);
		__m256 tmp2 = _mm256_mul_ps(guess, halfs);
		__m256 guess_new = _mm256_mul_ps(tmp1, tmp2);
		guess = _mm256_blendv_ps(guess, guess_new, mask);
		
		g2 = _mm256_mul_ps(guess, guess);
		error1 = _mm256_fmsub_ps(g2, vals, ones);
		error = _mm256_andnot_ps(abs, error1);

		mask = _mm256_cmp_ps(error, kThreshold, 30);
	}

        //output[i] = x * guess;
	__m256 res = _mm256_mul_ps(vals, guess);
	_mm256_storeu_ps(output + i, res);
    }
}

