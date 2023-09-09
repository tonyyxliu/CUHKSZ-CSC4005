/**
 * Retrieved from: https://enccs.github.io/OpenACC-CUDA-beginners/1.02_openacc-introduction/
 */

#include <stdio.h>
#ifdef _OPENACC
#include <openacc.h>
#endif

#define NX 102400

int main(void)
{
  double vecA[NX], vecB[NX], vecC[NX];
  double sum;
  int i;

  /* Initialization of the vectors */
  for (i = 0; i < NX; i++) {
    vecA[i] = 1.0 / ((double) (NX - i));
    vecB[i] = vecA[i] * vecA[i];
  }

#pragma acc data copy(vecA,vecB,vecC)
  {
#pragma acc parallel
    {
#pragma acc loop
      for (i = 0; i < NX; i++) {
        vecC[i] = vecA[i] * vecB[i];
      }
    }
  }
  sum = 0.0;
  /* Compute the check value */
  for (i = 0; i < NX; i++) {
    sum += vecC[i];
  }
  printf("Reduction sum: %18.16f\n", sum);

  return 0;
}
