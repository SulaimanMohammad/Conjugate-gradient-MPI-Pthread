#include "mpi_decomp.h"

#include <stdio.h>
#include <mpi.h>

void mpi_decomp_init(int ntot, mpi_decomp_t *mpi_info)
{
    int Q, R;

    MPI_Comm_rank(MPI_COMM_WORLD, &(mpi_info->mpi_rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(mpi_info->mpi_nproc));

    Q = ntot / mpi_info->mpi_nproc;
    R = ntot % mpi_info->mpi_nproc;

    mpi_info->mpi_ntot = ntot;

    if (mpi_info->mpi_rank < R) {

	mpi_info->mpi_nloc = Q+1;
	mpi_info->mpi_ideb = mpi_info->mpi_rank * (Q+1);
	mpi_info->mpi_ifin = mpi_info->mpi_ideb + mpi_info->mpi_nloc;

    } else {

	mpi_info->mpi_nloc = Q;
	mpi_info->mpi_ideb = R * (Q+1) + (mpi_info->mpi_rank - R) * Q;
	mpi_info->mpi_ifin = mpi_info->mpi_ideb + mpi_info->mpi_nloc;
    }

    printf("P%d/%d : ntot, nloc : %d, %d ; [ideb, ifin[ : [%d, %d[\n", 
	    mpi_info->mpi_rank, mpi_info->mpi_nproc,
	    mpi_info->mpi_ntot, mpi_info->mpi_nloc,
	    mpi_info->mpi_ideb, mpi_info->mpi_ifin);
}

