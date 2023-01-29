#ifndef __HYB_EXCHG_H
#define __HYB_EXCHG_H

#include <pthread.h>
#include <semaphore.h>
#include "mpi_decomp.h"

struct shared_exchg_s
{  
  int nthreads;
  double left_s;
  double right_s;

  pthread_mutex_t *exc_mutex;

  sem_t *semph;
  //sem_t *sem_master;
  int chose_th_master;
};
typedef struct shared_exchg_s shared_exchg_t;

/*
 * Initialisation/destruction d'une structure shared_exchg_t
 * nthreads : nombre de threads (du processus MPI) qui vont participer a l'echange
 */
void shared_exchg_init(shared_exchg_t *sh_ex, int nthreads);
void shared_exchg_destroy(shared_exchg_t *sh_ex);

/*
 * Echange hybride MPI/pthread
 * Si processus MPI existe "a gauche", lui envoie la valeur sh_arr[0] et recoit de lui *val_to_rcv_left
 * Si processus MPI existe "a droite", lui envoie la valeur sh_arr[mpi_decomp->mpi_nloc-1] et recoit de lui *val_to_rcv_right
 * Si processus voisin n'existe pas, valeur correspondante affectee a 0
 */
void hyb_exchg(
	double *sh_arr,
	shared_exchg_t *sh_ex,
	double *val_to_rcv_left, double *val_to_rcv_right,
	mpi_decomp_t *mpi_decomp);

#endif

