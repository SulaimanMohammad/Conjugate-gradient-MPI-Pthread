#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hyb_exchg.h"

#define NUM_WORKERS 4

struct args_exchg_s
{
    mpi_decomp_t *mpi_decomp;
    shared_exchg_t *sh_ex;
    double *sh_arr;
};
typedef struct args_exchg_s args_exchg_t;

double f(int i) {
    double xi = (double)i;
    return xi*cos(xi)/(1.+xi);
}

void *run_test_exchg(void *ptr_args_exchg)
{
    double val_to_rcv_left, val_to_rcv_right;
    double chck_val_to_rcv_left, chck_val_to_rcv_right;
    args_exchg_t *args_exchg = (args_exchg_t*)ptr_args_exchg;
    mpi_decomp_t *mpi_decomp = args_exchg->mpi_decomp;

    hyb_exchg(args_exchg->sh_arr, args_exchg->sh_ex, &val_to_rcv_left, &val_to_rcv_right, mpi_decomp);

    if (mpi_decomp->mpi_ideb == 0)
    {
        chck_val_to_rcv_left = 0.;
    }
    else
    {
        chck_val_to_rcv_left = f(mpi_decomp->mpi_ideb-1);

    }
    if (val_to_rcv_left != chck_val_to_rcv_left)
    {
        printf("P%d : val_to_rcv_left != chck_val_to_rcv_left : %.14e != %.14e\n",
                mpi_decomp->mpi_rank, val_to_rcv_left, chck_val_to_rcv_left);
    }

    if (mpi_decomp->mpi_ifin == mpi_decomp->mpi_ntot)
    {
        chck_val_to_rcv_right = 0.;
    }
    else
    {
        chck_val_to_rcv_right = f(mpi_decomp->mpi_ifin);
    }
    if (val_to_rcv_right != chck_val_to_rcv_right)
    {
        printf("P%d : val_to_rcv_right != chck_val_to_rcv_right : %.14e != %.14e\n",
                mpi_decomp->mpi_rank, val_to_rcv_right, chck_val_to_rcv_right);
    }

    return NULL;
}

int main(int argc, char **argv) {

    int Nglob, Nloc, mpi_thread_provided, t, i;
    double *x;
    mpi_decomp_t mpi_decomp;
    shared_exchg_t sh_ex;
    args_exchg_t args_ex[NUM_WORKERS];
    pthread_t pth_ex[NUM_WORKERS];

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_thread_provided);

    if (mpi_thread_provided != MPI_THREAD_SERIALIZED)
    {
        printf("Niveau demande' : MPI_THREAD_SERIALIZED, niveau fourni : %d\n", mpi_thread_provided);

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 1;
    }

    if (argc == 2) 
    {
        Nglob = atoi(argv[1]);
    }
    else
    {
        printf("Un seul argument obligatoire : %s <N> où <N> est la taille du tableau distribué entre processus\n", argv[0]);

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 1;
    }

    mpi_decomp_init(Nglob, &mpi_decomp);
    Nloc = mpi_decomp.mpi_nloc;

    x = (double*)malloc(Nloc*sizeof(double));

    for(i = mpi_decomp.mpi_ideb ; i < mpi_decomp.mpi_ifin ; i++)
    {
        x[i - mpi_decomp.mpi_ideb] = f(i);
    }

    shared_exchg_init(&sh_ex, NUM_WORKERS);

    for(t = 0 ; t < NUM_WORKERS ; t++)
    {
        args_ex[t].mpi_decomp = &mpi_decomp;
        args_ex[t].sh_ex      = &sh_ex;
        args_ex[t].sh_arr     = x;

        pthread_create(pth_ex+t, NULL, run_test_exchg, args_ex+t);
    }

    for(t = 0 ; t < NUM_WORKERS ; t++)
    {
        pthread_join(pth_ex[t], NULL);
    }

    free(x);

    shared_exchg_destroy(&sh_ex);

    MPI_Finalize();

    return 0;
}

