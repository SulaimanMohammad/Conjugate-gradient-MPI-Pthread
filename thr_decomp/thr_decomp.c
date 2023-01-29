#include "thr_decomp.h"

#include <stdio.h>

void thr_decomp_init(int ntot, int trank, int nthreads, thr_decomp_t *thr_info)
{
    int Q, R;

    thr_info->thr_rank     = trank;
    thr_info->thr_nthreads = nthreads;

    Q = ntot / thr_info->thr_nthreads;
    R = ntot % thr_info->thr_nthreads;

    thr_info->thr_ntot = ntot;

    if (thr_info->thr_rank < R) {

	thr_info->thr_nloc = Q+1;
	thr_info->thr_ideb = thr_info->thr_rank * (Q+1);
	thr_info->thr_ifin = thr_info->thr_ideb + thr_info->thr_nloc;

    } else {

	thr_info->thr_nloc = Q;
	thr_info->thr_ideb = R * (Q+1) + (thr_info->thr_rank - R) * Q;
	thr_info->thr_ifin = thr_info->thr_ideb + thr_info->thr_nloc;
    }

    printf("T%d/%d : ntot, nloc : %d, %d ; [ideb, ifin[ : [%d, %d[\n", 
	    thr_info->thr_rank, thr_info->thr_nthreads,
	    thr_info->thr_ntot, thr_info->thr_nloc,
	    thr_info->thr_ideb, thr_info->thr_ifin);
}

