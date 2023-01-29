#include "hyb_reduc.h"
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

void shared_reduc_init(shared_reduc_t *sh_red, int nthreads, int nvals)
{
   sh_red->nthreads = nthreads;
   sh_red->nvals = nvals;

   sh_red->th_mater=0;

   sh_red->red_val = malloc(sizeof(double) * nvals);

   sh_red->red_mutex = malloc(sizeof(pthread_mutex_t));
   pthread_mutex_init(sh_red->red_mutex, NULL);

   sh_red->red_barrier = malloc(sizeof(pthread_barrier_t));
   pthread_barrier_init(sh_red->red_barrier, NULL, nthreads);//the barrier counter is all the threads number to wait all

   sh_red->semph = malloc(sizeof(sem_t));
   sem_init(sh_red->semph, 0, 1);//acess for only one

   sh_red->semph_th_num = malloc(sizeof(sem_t));
   sem_init(sh_red->semph_th_num, 0, sh_red->nthreads);

}

void shared_reduc_destroy(shared_reduc_t *sh_red)
{
    pthread_barrier_destroy(sh_red->red_barrier);
    pthread_mutex_destroy(sh_red->red_mutex);
    sem_destroy(sh_red->semph);
    sem_destroy(sh_red->semph_th_num);

    free(sh_red->red_barrier);
    free(sh_red->red_mutex);
    free(sh_red->semph);
    free(sh_red->red_val);
    free(sh_red->semph_th_num);

}

/*
 * Reduction  hybride MPI/pthread
 * in  : tableau des valeurs a reduire (de dimension sh_red->nvals)
 * out : tableau des valeurs reduites  (de dimension sh_red->nvals)
 */
void hyb_reduc_sum(double *in, double *out, shared_reduc_t *sh_red)
{
    /*threads level */
    //reduction local
    int semVal;

sem_wait(sh_red->semph_th_num);

    pthread_mutex_lock(sh_red->red_mutex);

     for (int i = 0; i < sh_red->nvals; i++)
      {
        sh_red->red_val[i]= sh_red->red_val[i]+ in[i];
      }
    pthread_mutex_unlock(sh_red->red_mutex);

    pthread_barrier_wait(sh_red->red_barrier);// wait all the threads to the reductions


sem_wait(sh_red->semph);//give acess to just one thread

  if (sh_red->th_mater == 0) //chose first time
  {
      int root=0;
      int rank, size; // return the data of the communication
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);


      sh_red->th_mater = 1;

      double *buff = malloc(sizeof(double) * size);//buffer for the final data

      // gather all the data in the root process
      for (int i = 0; i < sh_red->nvals; i++)
        {
          /*int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)*/

          MPI_Gather(&(sh_red->red_val[i]), 1, MPI_DOUBLE, buff, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

          //the final result in the root
          if (rank == root)
            {
              for (int j = 0; j < size; j++)
                {
                  out[i] += buff[j];
                }
            }
        }

      free(buff);

      //int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
      //
      MPI_Bcast(out, sh_red->nvals, MPI_DOUBLE, root, MPI_COMM_WORLD);//senf the data we have to all the process to be used


      pthread_mutex_lock(sh_red->red_mutex);
      for (int i = 0; i < sh_red->nvals; i++)
        {
          sh_red->red_val[i] = out[i];
        }
      pthread_mutex_unlock(sh_red->red_mutex);

   }

   else //not master
   {
    pthread_mutex_lock(sh_red->red_mutex);

       for (int i = 0; i < sh_red->nvals; i++)
        {
          out[i] = sh_red->red_val[i];
        }
    pthread_mutex_unlock(sh_red->red_mutex);
   }

  sem_post(sh_red->semph);


  sem_getvalue(sh_red->semph_th_num, &semVal);
  if (semVal==0) //check if all threads executed the code
  {
      sh_red->th_mater=0;
  }

  MPI_Barrier(MPI_COMM_WORLD);//synch MPI

}
