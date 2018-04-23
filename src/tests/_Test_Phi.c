#include "fast_fm.h"
#include <fenv.h>
#include "TestFixtures.h"
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>



void test_als_weighted_warm_start(TestFixture_T *pFix) {
  int n_features = pFix->X->n;
  int n_samples = pFix->X->m;
  int k = 4;

  ffm_vector *y_10_iter = ffm_vector_calloc(n_samples);
  ffm_vector *y_15_iter = ffm_vector_calloc(n_samples);
  ffm_vector *y_5_plus_5_iter = ffm_vector_calloc(n_samples);
  
  //  C
  double data[n_samples];
  
  for ( int i = 0; i < n_samples; i++ ) {
     data[ i ] = 1; 
  }
  
  ffm_vector *C = ffm_vector_calloc(n_samples);
  C->size = n_samples;
  C->data = data;
  C->owner = 0;
  

  ffm_param param = {.warm_start = false,
                     .init_sigma = 0.1,
                     .SOLVER = SOLVER_ALS,
                     .TASK = TASK_REGRESSION,
                     .rng_seed = 123};

  param.n_iter = 10;
  ffm_coef *coef = alloc_fm_coef(n_features, k, false);
  sparse_fit_weighted(coef, pFix->X, NULL, pFix->y, NULL, param, C);
  sparse_predict(coef, pFix->X, y_10_iter);

  param.n_iter = 15;
  sparse_fit_weighted(coef, pFix->X, NULL, pFix->y, NULL, param, C);
  sparse_predict(coef, pFix->X, y_15_iter);

  param.n_iter = 5;
  sparse_fit_weighted(coef, pFix->X, NULL, pFix->y, NULL, param, C);
  param.warm_start = true;
  sparse_fit_weighted(coef, pFix->X, NULL, pFix->y, NULL, param, C);
  sparse_predict(coef, pFix->X, y_5_plus_5_iter);

  // check that the results are equal
  double mse = ffm_vector_mean_squared_error(y_10_iter, y_5_plus_5_iter);
  double mse_diff = ffm_vector_mean_squared_error(y_15_iter, y_5_plus_5_iter);

  printf("%f\n", mse);
  printf("%f\n", mse_diff);
  
  g_assert_cmpfloat(mse, <=, 1e-8);
  g_assert_cmpfloat(mse, <, mse_diff);

  free_ffm_coef(coef);
  ffm_vector_free_all(y_10_iter, y_5_plus_5_iter);
}



int main(int argc, char **argv){
  
  int n_features = 10;
  int n_samples = 10000;
  int k = 2;
  int seed = 15;

  TestFixture_T *pFix = makeTestFixture(seed, n_samples, n_features, k);
  
  
  test_als_weighted_warm_start(pFix);

}

