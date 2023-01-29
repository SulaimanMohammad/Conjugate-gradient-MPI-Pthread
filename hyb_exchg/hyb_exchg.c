#include "hyb_exchg.h"
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

/*
 * Initialisation/destruction d'une structure shared_exchg_t
 * nthreads : nombre de threads (du processus MPI) qui vont participer a l'echange
 */
void shared_exchg_init(shared_exchg_t *sh_ex, int nthreads)
{
    sh_ex->nthreads = nthreads;
    sh_ex->left_s =0; sh_ex->right_s = 0 ;

    sh_ex->semph = malloc(sizeof(sem_t));
    sem_init(sh_ex->semph, 1, 1);// smaphore shared between processes not only threads

    sh_ex->exc_mutex = malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(sh_ex->exc_mutex, NULL);

    sh_ex->chose_th_master=0;

}

void shared_exchg_destroy(shared_exchg_t *sh_ex)
{
  sem_destroy(sh_ex->semph);
  pthread_mutex_destroy(sh_ex->exc_mutex);

  free(sh_ex->semph);
  free(sh_ex->exc_mutex);

}


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
	mpi_decomp_t *mpi_decomp)
{

 sem_wait(sh_ex->semph);// one thread per time

  if (sh_ex->chose_th_master==0) //the first thread of a process 
   {
    // mpi_decom contains mpi_nproc number of processes
    // mpi_rank id of process
    // last process is  mpi_decomp->mpi_nproc - 1;

     if((mpi_decomp->mpi_nproc-1)== 0) //just one process then no left and right so nothing to do
       {
           printf("one process so nothing to echange\n");
           exit(0);
       }

      else if (mpi_decomp->mpi_rank == 0)//the first processes
      {
          // this process is on the extrem left so it cant receive from the left
          *val_to_rcv_left = 0.0;
          //Si processus MPI existe "à droite", lui envoie la valeur sh_arr[mpi_decomp->mpi_nloc-1] de lui

          //int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
          MPI_Ssend(&(sh_arr[mpi_decomp->mpi_nloc - 1]), 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

          //int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status * status)
          MPI_Recv(&(sh_ex->right_s), 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      }
      else if(mpi_decomp->mpi_rank==(mpi_decomp->mpi_nproc-1)) //the last process
      {
         // this process is on the extrem right  so it cant receive from the right
          *val_to_rcv_right = 0.0;

        //Si processus MPI existe "à gauche", lui envoie la valeur sh_arr[0]

        MPI_Recv(&(sh_ex->left_s), 1, MPI_DOUBLE, (mpi_decomp->mpi_rank - 1) , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Ssend(&(sh_arr[0]), 1, MPI_DOUBLE, (mpi_decomp->mpi_rank - 1) , 0, MPI_COMM_WORLD);

      }
      else // processes in between
      {
        //in between so they can receive from left and right
            // Recv from left
          MPI_Recv(&(sh_ex->left_s), 1, MPI_DOUBLE, (mpi_decomp->mpi_rank - 1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          // Send to right
          MPI_Ssend(&(sh_arr[mpi_decomp->mpi_nloc - 1]), 1,MPI_DOUBLE, (mpi_decomp->mpi_rank + 1), 0, MPI_COMM_WORLD);

          // Recv from right
          MPI_Recv(&(sh_ex->right_s), 1, MPI_DOUBLE, (mpi_decomp->mpi_rank + 1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          // Send to left
          MPI_Ssend(&(sh_arr[0]), 1, MPI_DOUBLE, (mpi_decomp->mpi_rank - 1), 0, MPI_COMM_WORLD);
      }

      //put the buffer of the message in the output pointer
      *val_to_rcv_right= sh_ex->right_s;
      *val_to_rcv_left= sh_ex->left_s;

       sh_ex->chose_th_master=1;
   }

   else // the rest of the threads
   {
      //shared for all the threads of MPI process
       pthread_mutex_lock(sh_ex->exc_mutex);

       *val_to_rcv_left = sh_ex->left_s;
       *val_to_rcv_right = sh_ex->right_s;

       pthread_mutex_unlock(sh_ex->exc_mutex);

   }
    sem_post(sh_ex->semph); //give place to the next thread
}
