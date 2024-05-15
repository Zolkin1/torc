#include <math.h>
#include <stdio.h>

typedef struct Array {
    void* data;
    unsigned long size;
    int sparse;
    const unsigned long* idx;
    unsigned long nnz;
} Array;

struct LangCAtomicFun {
    void* libModel;
    int (*forward)(void* libModel,
                   int atomicIndex,
                   int q,
                   int p,
                   const Array tx[],
                   Array* ty);
    int (*reverse)(void* libModel,
                   int atomicIndex,
                   int p,
                   const Array tx[],
                   Array* px,
                   const Array py[]);
};

void AutodiffCostInstance_jacobian(double const *const * in,
                                   double*const * out,
                                   struct LangCAtomicFun atomicFun) {
   //independent variables
   const double* x = in[0];

   //dependent variables
   double* jac = out[0];

   // auxiliary variables
   double v[1];

   v[0] = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11] + x[12] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19] + x[20] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26] + x[27] + x[28] + x[29] + x[30] + x[31] + x[32] + x[33] + x[34] + x[35] + x[36] + x[37] + x[38] + x[39] + x[40] + x[41] + x[42] + x[43] + x[44] + x[45] + x[46] + x[47] + x[48] + x[49];
   jac[0] = exp(sin(v[0])) * cos(v[0]);
   // variable duplicates: 49
   jac[1] = jac[0];
   jac[2] = jac[0];
   jac[3] = jac[0];
   jac[4] = jac[0];
   jac[5] = jac[0];
   jac[6] = jac[0];
   jac[7] = jac[0];
   jac[8] = jac[0];
   jac[9] = jac[0];
   jac[10] = jac[0];
   jac[11] = jac[0];
   jac[12] = jac[0];
   jac[13] = jac[0];
   jac[14] = jac[0];
   jac[15] = jac[0];
   jac[16] = jac[0];
   jac[17] = jac[0];
   jac[18] = jac[0];
   jac[19] = jac[0];
   jac[20] = jac[0];
   jac[21] = jac[0];
   jac[22] = jac[0];
   jac[23] = jac[0];
   jac[24] = jac[0];
   jac[25] = jac[0];
   jac[26] = jac[0];
   jac[27] = jac[0];
   jac[28] = jac[0];
   jac[29] = jac[0];
   jac[30] = jac[0];
   jac[31] = jac[0];
   jac[32] = jac[0];
   jac[33] = jac[0];
   jac[34] = jac[0];
   jac[35] = jac[0];
   jac[36] = jac[0];
   jac[37] = jac[0];
   jac[38] = jac[0];
   jac[39] = jac[0];
   jac[40] = jac[0];
   jac[41] = jac[0];
   jac[42] = jac[0];
   jac[43] = jac[0];
   jac[44] = jac[0];
   jac[45] = jac[0];
   jac[46] = jac[0];
   jac[47] = jac[0];
   jac[48] = jac[0];
   jac[49] = jac[0];
}

