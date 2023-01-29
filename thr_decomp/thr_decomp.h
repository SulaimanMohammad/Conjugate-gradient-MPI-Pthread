#ifndef __THR_DECOMP_H
#define __THR_DECOMP_H

/*
 * Decoupage d'un intervalle [0, thr_ntot[ entre thr_nthreads threads
 * Le thread thr_rank travaille sur l'intervalle [thr_ideb, thr_ifin[
 */
struct thr_decomp_s 
{
    int thr_nthreads;
    int thr_rank;

    int thr_ntot;
    int thr_nloc;

    int thr_ideb;
    int thr_ifin;
};
typedef struct thr_decomp_s thr_decomp_t;

/*
 * Remplissage des champs d'une struture thr_decomp_t
 */
void thr_decomp_init(int ntot, int trank, int nthreads, thr_decomp_t *thr_info);



#endif

