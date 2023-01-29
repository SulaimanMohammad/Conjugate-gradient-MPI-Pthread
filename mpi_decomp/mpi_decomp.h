#ifndef __MPI_DECOMP_H
#define __MPI_DECOMP_H

/*
 * Decoupage d'un intervalle [0, mpi_ntot[ entre les mpi_nproc processus MPI
 * Le processus mpi_rank travaille sur l'intervalle [mpi_ideb, mpi_ifin[
 */
struct mpi_decomp_s 
{
    int mpi_nproc;
    int mpi_rank;

    int mpi_ntot;
    int mpi_nloc;

    int mpi_ideb;
    int mpi_ifin;
};
typedef struct mpi_decomp_s mpi_decomp_t;

/*
 * Remplissage des champs d'une struture mpi_decomp_t
 */
void mpi_decomp_init(int ntot, mpi_decomp_t *mpi_info);

#endif

