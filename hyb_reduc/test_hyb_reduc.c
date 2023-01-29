#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <pthread.h>

#include "mpi_decomp.h"
#include "hyb_reduc.h"

#define NUM_WORKERS 4


struct args_test_s
{
    int trank;
    mpi_decomp_t   *mpi_decomp;
    shared_reduc_t *sh_red;
};
typedef struct args_test_s args_test_t;

void *run_test(void *ptr_args_test)
{
    args_test_t *args_test = (args_test_t*)ptr_args_test;
    int glob_rank, sum_glob, ntasks_glob, i;
    double in[2], out[2], chck_val[2], err;

    glob_rank = args_test->mpi_decomp->mpi_rank*NUM_WORKERS + args_test->trank;
    in[0] = (double)glob_rank;
    in[1] = -in[0];

    hyb_reduc_sum(in, out, args_test->sh_red);

    ntasks_glob = NUM_WORKERS*args_test->mpi_decomp->mpi_nproc;
    sum_glob = (ntasks_glob*(ntasks_glob-1))/2;
    chck_val[0] = (double)sum_glob;
    chck_val[1] = -chck_val[0];

    for(i = 0 ; i < 2 ; i++)
    {
	err = (out[i] - chck_val[i])/chck_val[i];

	if (fabs(err) < 1.e-14)
	{
	    if (glob_rank == 0)
	    {
		printf("Val reduite %d correcte\n", i);
	    }
	}
	else
	{
	    printf("P%d.T%d : out[%d] != chck_val[%d] : %.14e != %.14e, err = %.6e\n",
		    args_test->mpi_decomp->mpi_rank, args_test->trank,
		    i, i, out[i], chck_val[i], err);
	}
    }

    return NULL;
}

int main(int argc, char **argv) {

    int mpi_thread_provided, t;
    double *x, *y;
    double res, err;
    mpi_decomp_t mpi_decomp;
    shared_reduc_t sh_red;
    args_test_t args_test[NUM_WORKERS];
    pthread_t pth_test[NUM_WORKERS];

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_thread_provided);

    if (mpi_thread_provided != MPI_THREAD_SERIALIZED)
    {
	printf("Niveau demande' : MPI_THREAD_SERIALIZED, niveau fourni : %d\n", mpi_thread_provided);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 1;
    }

    mpi_decomp_init(10000, &mpi_decomp);


    shared_reduc_init(&sh_red, NUM_WORKERS, 2); /* 2 = deux valeurs a reduire */

    for(t = 0 ; t < NUM_WORKERS ; t++)
    {
	args_test[t].trank = t;
	args_test[t].mpi_decomp = &mpi_decomp;
	args_test[t].sh_red     = &sh_red;

	pthread_create(pth_test+t, NULL, run_test, args_test+t);
    }

    for(t = 0 ; t < NUM_WORKERS ; t++)
    {
	pthread_join(pth_test[t], NULL);
    }

    shared_reduc_destroy(&sh_red);

    MPI_Finalize();

    return 0;
}

