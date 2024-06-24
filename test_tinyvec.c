/*
 * MIT License
 * 
 * Copyright (c) 2024 Josh Hope-Collins
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

# include "tinyvec.h"

# include <assert.h>
# include <math.h>

/*
 * Macros for all the tests over different scalar types
 */

# define check(arg) assert(arg == 0)
# define close(abs, x, y, eps) assert(abs(x-y)<eps)

# define TINYVEC_TEST_FILL(p, dtype) \
int fill_##p##vec(dtype const    offset, \
                  dtype const      rate, \
                  p##vector *const  vec) \
{ \
   for( size_t i=0; i<(vec->n); ++i ){ (vec->data)[i] = offset+rate*i; } \
   return 0; \
} \

# define TINYVEC_TEST_COPY(p, dtype, xo, xr) \
int test_##p##vector_copy(size_t const n)\
{\
\
   p##vector *src;\
   check(p##vector_create(n, &src));\
   check(fill_##p##vec(xo, xr, src));\
\
   p##vector *dst;\
   check(p##vector_create(n, &dst));\
   check(p##vector_copy(src, dst));\
\
   for( size_t i=0; i<n; ++i ){ assert((dst->data)[i] == (src->data)[i]); }\
\
   check(p##vector_destroy(&src));\
   check(p##vector_destroy(&dst));\
\
   return 0;\
}\

# define TINYVEC_TEST_DUPLICATE(p, dtype, xo, xr) \
int test_##p##vector_duplicate(size_t const n)\
{\
   /* initialise some source data */\
   p##vector *src;\
   check(p##vector_create(n, &src));\
   check(fill_##p##vec(xo, xr, src));\
\
   /* copy it over to the destination */\
   p##vector *dst;\
   check(p##vector_duplicate(src, &dst));\
\
   /* check the copy is correct */\
   for( size_t i=0; i<n; ++i ){ assert((dst->data)[i] == (src->data)[i]); }\
\
   /* clean up */\
   check(p##vector_destroy(&src));\
   check(p##vector_destroy(&dst));\
\
   return 0;\
}\

# define TINYVEC_TEST_AXPY(p, dtype, eps, abs, a, b, xo, xr, yo, yr) \
int test_##p##vector_axpy(size_t const n)\
{\
   p##vector *x;\
   p##vector *y;\
   check(p##vector_create(n, &x));\
   check(p##vector_create(n, &y));\
\
   fill_##p##vec(xo, xr, x);\
   fill_##p##vec(yo, yr, y);\
\
   p##vector_axpy(a, x, y);\
\
   for( size_t i=0; i<n; ++i )\
  {\
      const dtype expected = a*(xo+xr*i) + (yo+i*yr);\
      close(abs,(y->data)[i], expected, eps);\
  }\
\
   check(p##vector_destroy(&x));\
   check(p##vector_destroy(&y));\
\
   return 0;\
}\

# define TINYVEC_TEST_AYPX(p, dtype, eps, abs, a, b, xo, xr, yo, yr) \
int test_##p##vector_aypx(size_t const n)\
{\
   p##vector *x;\
   p##vector *y;\
   check(p##vector_create(n, &x));\
   check(p##vector_create(n, &y));\
\
   fill_##p##vec(xo, xr, x);\
   fill_##p##vec(yo, yr, y);\
\
   p##vector_aypx(a, x, y);\
\
   for( size_t i=0; i<n; ++i )\
  {\
      const dtype expected = (xo+xr*i) + a*(yo+i*yr);\
      close(abs, (y->data)[i], expected, eps);\
  }\
\
   check(p##vector_destroy(&x));\
   check(p##vector_destroy(&y));\
\
   return 0;\
}\

# define TINYVEC_TEST_AXPBY(p, dtype, eps, abs, a, b, xo, xr, yo, yr) \
int test_##p##vector_axpby(size_t const n)\
{\
   p##vector *x;\
   p##vector *y;\
   check(p##vector_create(n, &x));\
   check(p##vector_create(n, &y));\
\
   fill_##p##vec(xo, xr, x);\
   fill_##p##vec(yo, yr, y);\
\
   p##vector_axpby(a, x, b, y);\
\
   for( size_t i=0; i<n; ++i )\
  {\
      const dtype expected = a*(xo+xr*i) + b*(yo+i*yr);\
      close(abs, (y->data)[i], expected, eps);\
  }\
\
   check(p##vector_destroy(&x));\
   check(p##vector_destroy(&y));\
\
   return 0;\
}\

# define TINYVEC_TEST_WAXPY(p, dtype, eps, abs, a, b, xo, xr, yo, yr) \
int test_##p##vector_waxpy(size_t const n)\
{\
   p##vector *x;\
   p##vector *y;\
   p##vector *w;\
   check(p##vector_create(n, &x));\
   check(p##vector_create(n, &y));\
   check(p##vector_create(n, &w));\
\
   fill_##p##vec(xo, xr, x);\
   fill_##p##vec(yo, yr, y);\
\
   p##vector_waxpy(w, a, x, y);\
\
   for( size_t i=0; i<n; ++i )\
  {\
      const dtype expected = a*(xo+xr*i) + (yo+i*yr);\
      close(abs, (w->data)[i], expected, eps);\
  }\
\
   check(p##vector_destroy(&x));\
   check(p##vector_destroy(&y));\
   check(p##vector_destroy(&w));\
\
   return 0;\
}\

# define TINYVEC_TEST_WAXPBY(p, dtype, eps, abs, a, b, xo, xr, yo, yr) \
int test_##p##vector_waxpby(size_t const n)\
{\
   p##vector *x;\
   p##vector *y;\
   p##vector *w;\
   check(p##vector_create(n, &x));\
   check(p##vector_create(n, &y));\
   check(p##vector_create(n, &w));\
\
   fill_##p##vec(xo, xr, x);\
   fill_##p##vec(yo, yr, y);\
\
   p##vector_waxpby(w, a, x, b, y);\
\
   for( size_t i=0; i<n; ++i )\
  {\
      const dtype expected = a*(xo+xr*i) + b*(yo+i*yr);\
      close(abs, (w->data)[i], expected, eps);\
  }\
\
   check(p##vector_destroy(&x));\
   check(p##vector_destroy(&y));\
   check(p##vector_destroy(&w));\
\
   return 0;\
}\

# define TINYVEC_TEST_NORM(p, dtype, eps, abs) \
int test_##p##vector_norm(size_t const n)\
{\
   p##vector *x;\
   check(p##vector_create(n, &x));\
\
   fill_##p##vec(0, 1, x);\
   dtype expected = 0;\
   for( size_t i=0; i<n; ++i ){ expected+=i*i; }\
   expected = sqrt(expected);\
\
   const dtype norm = p##vector_norm(x);\
\
   close(abs, norm, expected, eps);\
\
   check(p##vector_destroy(&x));\
\
   return 0;\
}

/*
 * define all tests for a scalar type together
 */
# define GENERATE_TINYVEC_TESTS(p, dtype, a, b, xo, xr, yo, yr, eps, abs) \
TINYVEC_TEST_FILL(     p, dtype) \
TINYVEC_TEST_COPY(     p, dtype, xo, xr) \
TINYVEC_TEST_DUPLICATE(p, dtype, xo, xr) \
TINYVEC_TEST_AXPY(     p, dtype, eps, abs, a, b, xo, xr, yo, yr) \
TINYVEC_TEST_AYPX(     p, dtype, eps, abs, a, b, xo, xr, yo, yr) \
TINYVEC_TEST_AXPBY(    p, dtype, eps, abs, a, b, xo, xr, yo, yr) \
TINYVEC_TEST_WAXPY(    p, dtype, eps, abs, a, b, xo, xr, yo, yr) \
TINYVEC_TEST_WAXPBY(   p, dtype, eps, abs, a, b, xo, xr, yo, yr) \
TINYVEC_TEST_NORM(     p, dtype, eps, abs)


/*
 * Concrete tests
 */
GENERATE_TINYVEC_TESTS(d, double, 4.24, 2.17, -4.23, 2.27, 1.80, -0.84, sqrt(1e-16), fabs)
GENERATE_TINYVEC_TESTS(f, float,  4.24, 2.17, -4.23, 2.27, 1.80, -0.84, sqrt(1e-8), fabsf)

# ifdef TINYVEC_COMPLEX
GENERATE_TINYVEC_TESTS(cd, complex double, (4.2-0.8*I), (2.7+0.8*I), (-4.2-1.4*I), (2.7-2.6*I), (1.8+0.3*I), (-0.8+0.2*I), sqrt(1e-16), cabs)
GENERATE_TINYVEC_TESTS(cf, complex float,  (4.2-0.8*I), (2.7+0.8*I), (-4.2-1.4*I), (2.7-2.6*I), (1.8+0.3*I), (-0.8+0.2*I), sqrt(1e-8), cabsf)
# endif


/*
 * Run the lot
 */
int main()
{
   size_t const n=16;

   check(test_dvector_copy(n));
   check(test_dvector_axpy(n));
   check(test_dvector_aypx(n));
   check(test_dvector_axpby(n));
   check(test_dvector_waxpy(n));
   check(test_dvector_waxpby(n));
   check(test_dvector_norm(n));

   check(test_fvector_copy(n));
   check(test_fvector_axpy(n));
   check(test_fvector_aypx(n));
   check(test_fvector_axpby(n));
   check(test_fvector_waxpy(n));
   check(test_fvector_waxpby(n));
   check(test_fvector_norm(n));

# ifdef TINYVEC_COMPLEX
   check(test_cdvector_copy(n));
   check(test_cdvector_axpy(n));
   check(test_cdvector_aypx(n));
   check(test_cdvector_axpby(n));
   check(test_cdvector_waxpy(n));
   check(test_cdvector_waxpby(n));
   check(test_cdvector_norm(n));

   check(test_cfvector_copy(n));
   check(test_cfvector_axpy(n));
   check(test_cfvector_aypx(n));
   check(test_cfvector_axpby(n));
   check(test_cfvector_waxpy(n));
   check(test_cfvector_waxpby(n));
   check(test_cfvector_norm(n));
# endif
   return 0;
}
