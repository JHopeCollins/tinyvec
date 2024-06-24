/*
 * A tiny fixed size vector struct.
 *
 * Vector:
 *   - knows it's own size
 *   - has standard arithmetic operations
 *   - is implemented for real and complex scalar types
 */


# ifndef _TINY_HEADER
# define _TINY_HEADER

# include <stdlib.h>
# include <math.h>

# ifdef TINYVEC_DEBUG
   # include <assert.h>
   # define check_vectors(x, y) assert(x->n == y->n)
# else
   # define check_vectors(x, y)
# endif

# ifdef TINYVEC_COMPLEX
   # include <complex.h>
# endif


/*
 * Identity function alternative to conjugate for real scalars
 */
static inline double _tinyvec_idd(const double x){ return x; }
static inline float  _tinyvec_idf(const float  x){ return x; }


/*
 * Vector struct for double precision data
 */
# define STRUCT(p, dtype) \
typedef struct _##p##vector \
{ \
   size_t     n; \
   dtype *data; \
} \
p##vector;


/*
 * Create vector of length n
 */
# define CREATE(p, dtype) \
static inline  \
int p##vector_create(const size_t n, \
               p##vector**   vec) \
{ \
   (*vec) = (p##vector*)malloc(sizeof(p##vector)); \
   if((*vec)==NULL){ free(*vec); return 1; } \
 \
   (*vec)->data = (dtype*)malloc(n*sizeof(dtype)); \
   if (((*vec)->data)==NULL) \
   { \
      free((*vec)->data); \
      free(*vec); \
      return 1; \
   } \
   (*vec)->n = n; \
   return 0; \
}


/*
 * Free vector
 */
# define DESTROY(p, dtype) \
static inline  \
int p##vector_destroy(p##vector** vec) \
{ \
   if(((*vec)->data) != NULL){ free((*vec)->data); } \
   if((*vec) != NULL){ free(*vec); } \
   return 0; \
}


/*
 * Copy data from one vector to an existing vector
 */
# define COPY(p, dtype) \
static inline  \
int p##vector_copy(p##vector const *const src, \
                   p##vector *const       dst) \
{ \
   check_vectors(src, dst); \
   for( size_t i=0; i<(src->n); ++i ){ (dst->data)[i] = (src->data)[i]; } \
   return 0; \
}


/*
 * Create a new vector by duplicating an existing vector
 */
# define DUPLICATE(p, dtype) \
static inline  \
int p##vector_duplicate(p##vector const *const src, \
                        p##vector**            dst) \
{ \
   int err = 0; \
   err = p##vector_create(src->n, dst); if( err != 0 ){ return err; } \
   err = p##vector_copy(src, *dst); if( err != 0 ){ return err; } \
   return 0; \
}


/*
 * y = a*x + y
 */
# define AXPY(p, dtype) \
static inline  \
int p##vector_axpy(dtype const            a, \
                   p##vector const *const x, \
                   p##vector *const       y) \
{ \
   check_vectors(x, y); \
   for( size_t i=0; i<(x->n); ++i ) \
   { \
      (y->data)[i] =  a*(x->data)[i] + (y->data)[i]; \
   } \
   return 0; \
}


/*
 * y = x + a*y
 */
# define AYPX(p, dtype) \
static inline  \
int p##vector_aypx(dtype const            a, \
                   p##vector const *const x, \
                   p##vector *const       y) \
{ \
   check_vectors(x, y); \
   for( size_t i=0; i<(x->n); ++i ) \
   { \
      (y->data)[i] =  (x->data)[i] + a*(y->data)[i]; \
   } \
   return 0; \
}


/*
 * y = a*x + b*y
 */
# define AXPBY(p, dtype) \
static inline  \
int p##vector_axpby(dtype const            a, \
                    p##vector const *const x, \
                    dtype const            b, \
                    p##vector *const       y) \
{ \
   check_vectors(x, y); \
   for( size_t i=0; i<(x->n); ++i ) \
   { \
      (y->data)[i] =  a*(x->data)[i] + b*(y->data)[i]; \
   } \
   return 0; \
}


/*
 * w = a*x + y
 */
# define WAXPY(p, dtype) \
static inline  \
int p##vector_waxpy(p##vector *const       w, \
                    dtype const            a, \
                    p##vector const *const x, \
                    p##vector const *const y) \
{ \
   check_vectors(x, y); \
   check_vectors(x, w); \
   for( size_t i=0; i<(x->n); ++i ) \
   { \
      (w->data)[i] =  a*(x->data)[i] + (y->data)[i]; \
   } \
   return 0; \
}


/*
 * w = a*x + b*y
 */
# define WAXPBY(p, dtype) \
static inline  \
int p##vector_waxpby(p##vector *const       w, \
                     dtype const            a, \
                     p##vector const *const x, \
                     dtype const            b, \
                     p##vector const *const y) \
{ \
   check_vectors(x, y); \
   check_vectors(x, w); \
   for( size_t i=0; i<(x->n); ++i ) \
   { \
      (w->data)[i] =  a*(x->data)[i] + b*(y->data)[i]; \
   } \
   return 0; \
}


/*
 * Inner product of two p##vector
 */
# define INNER(p, dtype, conj) \
static inline  \
dtype p##vector_inner(p##vector const *const x, \
                      p##vector const *const y) \
{ \
   check_vectors(x, y); \
   dtype l = 0; \
   for( size_t i=0; i<(x->n); ++i ){ l+=(x->data)[i]*conj((x->data)[i]); } \
   return l; \
}


/*
 * L2 norm of p##vector
 */
# define NORM(p, dtype) \
static inline  \
dtype p##vector_norm(p##vector const *const x) \
{ \
   return sqrt(p##vector_inner(x, x)); \
}


# define GENERATE_TINYVEC(p, dtype, conj) \
STRUCT(   p, dtype) \
CREATE(   p, dtype) \
DESTROY(  p, dtype) \
COPY(     p, dtype) \
DUPLICATE(p, dtype) \
AXPY(     p, dtype) \
AYPX(     p, dtype) \
AXPBY(    p, dtype) \
WAXPY(    p, dtype) \
WAXPBY(   p, dtype) \
INNER(    p, dtype, conj) \
NORM(     p, dtype)

GENERATE_TINYVEC(d, double, _tinyvec_idd)
GENERATE_TINYVEC(f, float,  _tinyvec_idf)
# ifdef TINYVEC_COMPLEX
GENERATE_TINYVEC(cd, complex double, conj)
GENERATE_TINYVEC(cf, complex float, conjf)
# endif

# undef check_vectors
# undef STRUCT
# undef CREATE
# undef DESTROY
# undef COPY
# undef DUPLICATE
# undef AXPY
# undef AYPX
# undef AXPBY
# undef WAXPY
# undef WAXPBY
# undef INNER
# undef NORM
# undef GENERATE_TINYVEC

# endif
