
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <libgen.h>
#include "mpi_decomp.h"
#include "thr_decomp.h"
#include "hyb_reduc.h"
#include "hyb_exchg.h"
#include <mpi.h>

#define NUM_WORKERS 4
//x solution  vectore
//b right hand vctore
//A system matrix dimension N  this matrix should be devided and use MPI
//we need to do matrix*vectore and scal*vectores

//matrix 3-band  structure
struct matrix3b_s
{
    int N;/* Matrice de dimension NxN */

    /* Pour la ligne i,
     * A(i, i-1) = bnd[0][i]
     * A(i, i)   = bnd[1][i]
     * A(i, i+1) = bnd[2][i]
     * Tous les elements sur les colonnes autres que i-1, i et i+1 sont nuls
     */
    double *bnd[3];
};
typedef struct matrix3b_s matrix3b_t;

struct vector_s
{
  int N;          /* Vecteur de dimension N */
  double *elt;    /* elt[i] : i-ieme element du vecteur*/
};
typedef struct vector_s vector_t;

struct hyper_grad_conj
{
  matrix3b_t *A;
  vector_t *vb;
  vector_t *vx;

  mpi_decomp_t *mpi_decomp;
  thr_decomp_t *thr_decomp;
  shared_exchg_t *sh_ex;
  shared_reduc_t *sh_red;
};
typedef struct hyper_grad_conj hyper_gc;


void vector_alloc(int N, vector_t *vec)
{
  vec->N = N;
  vec->elt = (double*)malloc(N*sizeof(double));
}

void vector_free(vector_t *vec)
{
    free(vec->elt);
}

/*............................................................*/
/*............operations on vectors for each thread ..........*/
/*............................................................*/

//Initialise vector as 0, but here  it should be for each threead

void vector_init_0(hyper_gc *hyper_conj, vector_t *vec)
{
  for(int i = hyper_conj->thr_decomp->thr_ideb ; i < hyper_conj->thr_decomp->thr_ifin; i++)
  {
	 vec->elt[i] = 0.;
  }
}

// multiplication by scalar for each thread
void vector_mul_scal(hyper_gc *hyper_conj ,vector_t *vec, double s)
{
  for(int i = hyper_conj->thr_decomp->thr_ideb ; i < hyper_conj->thr_decomp->thr_ifin ; i++)
  {
	  vec->elt[i] *= s;
  }
}

// Assigning  vector by another multiply by a scalar for each thread
void vector_affect_mul_scal(hyper_gc *hyper_conj ,vector_t *vec_out, double s, vector_t *vec_in)
{
  assert(vec_out->N == vec_in->N);

  for(int i = hyper_conj->thr_decomp->thr_ideb ; i < hyper_conj->thr_decomp->thr_ifin ; i++)
  {
    vec_out->elt[i] = s*vec_in->elt[i];
  }
}

//compute norm L2 for threads  "  (|| vec ||_2)²  "
// the vectors are in threads to have the norm2 for all we need to use the reduction
double vector_norm2(hyper_gc *hyper_conj , vector_t *vec)
{
  double norm_th=0.;
  double norm2=0.;

  for(int i = hyper_conj->thr_decomp->thr_ideb ; i < hyper_conj->thr_decomp->thr_ifin ; i++)
  {
    norm_th += vec->elt[i]*vec->elt[i];
  }
  //void hyb_reduc_sum(double *in, double *out, shared_reduc_t *sh_red);
  hyb_reduc_sum(&norm_th, &norm2, hyper_conj->sh_red);

  return norm2;
}

//Add to a vector another vector multiplied by a scalar and that in each thread
void vector_add_mul_scal(hyper_gc *hyper_conj, vector_t *vec_inout, double s, vector_t *vec_in)
{
  assert(vec_inout->N == vec_in->N);

  for(int i = hyper_conj->thr_decomp->thr_ideb ; i < hyper_conj->thr_decomp->thr_ifin ; i++)
  {
	 vec_inout->elt[i] += s*vec_in->elt[i];
  }
}

//Returns the ratio of 2 dot products
//again here we need to use reduction for the total
double div_bi_prod_scal(hyper_gc *hyper_conj ,vector_t *v1, vector_t *w1, vector_t *v2, vector_t *w2)
{
  assert(v1->N == w1->N);
  assert(v1->N == v2->N);
  assert(v1->N == w2->N);

  double scal_th1, scal_th2, scal1, scal2 ;
  scal_th1= scal_th2=  scal1 =scal2=  0.;

  for(int i = hyper_conj->thr_decomp->thr_ideb ; i < hyper_conj->thr_decomp->thr_ifin ; i++)
  {
	  scal_th1 += v1->elt[i]*w1->elt[i];
	  scal_th2 += v2->elt[i]*w2->elt[i];
  }

  hyb_reduc_sum(&scal_th1, &scal1, hyper_conj->sh_red);
  hyb_reduc_sum(&scal_th2, &scal2, hyper_conj->sh_red);

  return scal1/scal2;
}
/*............................................................*/
/*............................................................*/

void linear_system_alloc_and_init(mpi_decomp_t *mpi_decomp, matrix3b_t *A, vector_t *vb)
{
  int N=mpi_decomp->mpi_nloc;
  // printf("n= %d\n",N);
  assert(N > 2);
  /* Allocations */
  A->N = N;
  A->bnd[0] = (double*)malloc(N*sizeof(double));
  A->bnd[1] = (double*)malloc(N*sizeof(double));
  A->bnd[2] = (double*)malloc(N*sizeof(double));

  vb->N   = N;
  vb->elt = (double*)malloc(N*sizeof(double));

  double coeff = 0.01;

  for(int i = 0 ; i < N ; i++)
  {
   A->bnd[0][i] = -coeff;
   A->bnd[1][i] = 1. + 2*coeff;
   A->bnd[2][i] = -coeff;
   vb->elt[i] = 1.;
  }

  //we have multiprocesses then processes deal with it
  //we chose the first one and the last one
  if(mpi_decomp->mpi_rank ==0)
  {
    A->bnd[0][0] = 0.;
    vb->elt[0] = 1. + coeff;
  }

  if(mpi_decomp->mpi_rank == mpi_decomp->mpi_nproc-1)
  {
    A->bnd[2][N-1] = 0.;
    vb->elt[N-1] = 1. + coeff;
  }
}

void linear_system_free(matrix3b_t *A, vector_t *vb)
{
  free(A->bnd[0]);
  free(A->bnd[1]);
  free(A->bnd[2]);
  free(vb->elt);
}

void prod_mat_vec( hyper_gc *hyper_conj, vector_t *vy, matrix3b_t *A, vector_t *vx)
{
  assert(A->N == vx->N);
  assert(vy->N == vx->N);
  int i;
  double val_left, val_right;
  hyb_exchg(vx->elt, hyper_conj->sh_ex, &val_left, &val_right, hyper_conj->mpi_decomp);
  // i think here we have many MPI processes each one has many threads
  //so the product will be based on processes and threads conditions

  //first MPI process and there are sub_threads
  //use thr_rank from thr_decomp
  if(hyper_conj->mpi_decomp->mpi_rank==0)
   {
      if(hyper_conj->thr_decomp->thr_rank==0) //nothing to exchange for first thread
      {
        i = 0;
        vy->elt[i] =
        A->bnd[1][i] * vx->elt[i] +
        A->bnd[2][i] * vx->elt[i+1];
      }
      else if(hyper_conj->thr_decomp->thr_rank == hyper_conj->thr_decomp->thr_nthreads-1)//last thread of first process with value of the right
      {
        i= hyper_conj->thr_decomp->thr_ifin - 1; //(N-1)
        vy->elt[i] =
      	A->bnd[0][i] * vx->elt[i-1] +
      	A->bnd[1][i] * val_right;
      }
      else //the rest of threads
      {
        for(i = hyper_conj->thr_decomp->thr_ideb ; i < hyper_conj->thr_decomp->thr_ifin ; i++)
        {
          vy->elt[i] =
          A->bnd[0][i] * vx->elt[i-1] +
          A->bnd[1][i] * vx->elt[i] +
          A->bnd[2][i] * vx->elt[i+1];
        }
      }
    }
  //last MPI process
  else if(hyper_conj->mpi_decomp->mpi_rank == hyper_conj->mpi_decomp-> mpi_nproc - 1)
    {
      if( hyper_conj->thr_decomp->thr_rank ==0) //last mpi will need the left value in the first thread
      {
        i = 0;
        vy->elt[i] =
        A->bnd[1][i] * val_left +
        A->bnd[2][i] * vx->elt[i+1];
      }
      else if(hyper_conj->thr_decomp->thr_rank == hyper_conj->thr_decomp->thr_nthreads-1)//last thread of first process with value of the right
      {
        i= hyper_conj->thr_decomp->thr_ifin - 1; //(N-1)
        vy->elt[i] =
        A->bnd[0][i] * vx->elt[i-1] +
        A->bnd[1][i] * vx->elt[i];
      }
      else //the rest of threads
      {
        for(i = hyper_conj->thr_decomp->thr_ideb ; i < hyper_conj->thr_decomp->thr_ifin ; i++)
        {
          vy->elt[i] =
          A->bnd[0][i] * vx->elt[i-1] +
          A->bnd[1][i] * vx->elt[i] +
          A->bnd[2][i] * vx->elt[i+1];
        }
      }
    }

  else//processes in between
    {
      if(hyper_conj->thr_decomp->thr_rank ==0) //last mpi will need the left value
      {
        i = 0;
        vy->elt[i] =
        A->bnd[0][i] * val_left +
        A->bnd[1][i] * vx->elt[i] +
        A->bnd[2][i] * vx->elt[i+1];
      }
      else if(hyper_conj->thr_decomp->thr_rank == hyper_conj->thr_decomp->thr_nthreads-1)
      {
        i= hyper_conj->thr_decomp->thr_ifin - 1; //(N-1)
        vy->elt[i] =
      	A->bnd[0][i] * vx->elt[i-1] +
      	A->bnd[1][i] * val_right;
      }
      else//threads in between
      {
        for(i = hyper_conj->thr_decomp->thr_ideb ; i < hyper_conj->thr_decomp->thr_ifin ; i++)
        {
          vy->elt[i] =
          A->bnd[0][i] * vx->elt[i-1] +
          A->bnd[1][i] * vx->elt[i] +
          A->bnd[2][i] * vx->elt[i+1];
        }
      }
    }
    pthread_barrier_wait(hyper_conj->sh_red->red_barrier);//synch threads , red_barrier from hyb_reduc.h
    MPI_Barrier(MPI_COMM_WORLD); //synch
}

//the algo
////threads routine  so it is pointer function and the old argumentin the sequential code are arg here
void *gradient_conjugue(void *arg)
{
  hyper_gc *hyper_conj=(  hyper_gc *)arg;
  vector_t *vb = hyper_conj->vb;
  vector_t *vx = hyper_conj->vx;

  matrix3b_t *A = hyper_conj->A;
  vector_t vg, vh, vw;
  double sn, sn1, sr, sg, seps;
  int k, N;

  assert(A->N == vb->N);
  assert(A->N == vx->N);

  seps = 1.e-12;
  N = A->N;
  vector_alloc(N, &vg);
  vector_alloc(N, &vh);
  vector_alloc(N, &vw);

 //Initialise
 vector_init_0(hyper_conj,vx);
 vector_affect_mul_scal(hyper_conj, &vg, -1., vb);
 vector_affect_mul_scal(hyper_conj ,&vh, -1., &vg);
 sn = vector_norm2(hyper_conj, &vg);

  //Iterative
  for(k = 0 ; k < N && sn > seps ; k++)
  {
   printf("Iteration %5d, err = %.4e\n", k, sn);
   prod_mat_vec(hyper_conj, &vw, A, &vh);

   sr = - div_bi_prod_scal(hyper_conj, &vg, &vh, &vh, &vw);

   vector_add_mul_scal(hyper_conj, vx, sr, &vh);
   vector_add_mul_scal(hyper_conj, &vg, sr, &vw);

   sn1 = vector_norm2(hyper_conj, &vg);

   sg = sn1 / sn;
   sn = sn1;

   vector_mul_scal(hyper_conj, &vh, sg);

   vector_add_mul_scal(hyper_conj, &vh, -1., &vg);
  }

  vector_free(&vg);
  vector_free(&vh);
  vector_free(&vw);

}

/// Check the result
//after computing  vx then  A.vx should be (vb known )

void *verif_sol(void *arg)
{
  hyper_gc *hyper_conj=(  hyper_gc *)arg;
  vector_t *vb = hyper_conj->vb;
  vector_t *vx = hyper_conj->vx;
  matrix3b_t *A = hyper_conj->A;
  vector_t vb_cal;
  double norm2;

  assert(A->N == vb->N);
  assert(A->N == vx->N);

  vector_alloc(A->N, &vb_cal);

  prod_mat_vec(hyper_conj, &vb_cal, A, vx); /* vb_cal = A.vx */
  vector_add_mul_scal(hyper_conj, &vb_cal, -1., vb); /* vb_cal = vb_cal - vb */
  norm2 = vector_norm2(hyper_conj ,&vb_cal);

  if (norm2 < 1.e-12)
  {
	 printf("Resolution correcte du systeme\n");
  }
  else
  {
	 printf("Resolution incorrecte du systeme, erreur : %.4e\n", norm2);
  }
}


int main(int argc, char **argv)
{
  int N;
  vector_t vx, vb;
  matrix3b_t A;// band matrix
  int mpi_thread_provided, rank, size;

  thr_decomp_t thr_decomp[NUM_WORKERS];//for each thread
  mpi_decomp_t mpi_decomp;//of mpi_decomp_t(mpi_decomp.h)
  shared_exchg_t sh_ex;// of shared_exchg_t (hyb_exchg.h)
  shared_reduc_t sh_red;
  pthread_t hyper_th[NUM_WORKERS];
  hyper_gc hyper_conj[NUM_WORKERS];// structure for each thread


  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_thread_provided);

  if (mpi_thread_provided != MPI_THREAD_SERIALIZED)
  {
    printf("Niveau demande' : MPI_THREAD_SERIALIZED, niveau fourni : %d\n", mpi_thread_provided);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 1;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  //take the dim

  if (argc != 2)
  {
    printf("Usage : %s <N>\n", basename(argv[0]));
    printf("\t<N> : dimension de la matrice\n");
    abort();
  }

  N = atoi(argv[1]);

  //decompse MPI
   mpi_decomp_init(N, &mpi_decomp);//return the structure in mpi_decomp
   //printf("mpi_nloc=%d\n", mpi_decomp.mpi_nloc);

  shared_exchg_init(&sh_ex, NUM_WORKERS);
  shared_reduc_init(&sh_red, NUM_WORKERS, 2); //reducr tow values

  //creat the system
  linear_system_alloc_and_init(&mpi_decomp, &A, &vb);
  vector_alloc(mpi_decomp.mpi_nloc, &vx);

 // call the function of the algo Gradient Conjugate for each thread
 for(int i=0 ; i<NUM_WORKERS; i++) //data for each thread
 {
   thr_decomp_init(mpi_decomp.mpi_nloc, i , NUM_WORKERS, &(thr_decomp[i]));
   hyper_conj[i].mpi_decomp=&mpi_decomp;
   hyper_conj[i].thr_decomp=&(thr_decomp[i]);
   hyper_conj[i].sh_ex=&sh_ex;
   hyper_conj[i].sh_red=&sh_red;
   hyper_conj[i].vx=&vx;
   hyper_conj[i].vb=&vb;
   hyper_conj[i].A=&A;
   if (pthread_create(hyper_th + i, NULL,gradient_conjugue, &(hyper_conj[i])) != 0)
   {
     perror("Failed to create thread");
   }
 }

 //wait the threads to be finished
 for(int i=0; i<NUM_WORKERS;i++)
 {
   if(pthread_join(hyper_th[i], NULL )!=0)
   {
    perror("Failed to join thread");
   }
 }

 //check here we should call verif_sol

 for(int i=0 ; i<NUM_WORKERS; i++) //data for each thread
 {
   thr_decomp_init(mpi_decomp.mpi_nloc, i , NUM_WORKERS, &(thr_decomp[i]));
   hyper_conj[i].mpi_decomp=&mpi_decomp;
   hyper_conj[i].thr_decomp=&(thr_decomp[i]);
   hyper_conj[i].sh_ex=&sh_ex;
   hyper_conj[i].sh_red=&sh_red;
   hyper_conj[i].vx=&vx;
   hyper_conj[i].vb=&vb;
   hyper_conj[i].A=&A;
   if (pthread_create(hyper_th + i, NULL,verif_sol, &(hyper_conj[i])) != 0)
   {
     perror("Failed to create thread");
   }
 }

 for(int i=0; i<NUM_WORKERS;i++)
 {
   if(pthread_join(hyper_th[i], NULL )!=0)
   {
    perror("Failed to join thread");
   }
 }
 //free and distroy
 linear_system_free(&A,&vb);
 vector_free(&vx);
 shared_reduc_destroy(&sh_red);
 shared_exchg_destroy(&sh_ex);

 MPI_Barrier(MPI_COMM_WORLD);
 MPI_Finalize();

}
