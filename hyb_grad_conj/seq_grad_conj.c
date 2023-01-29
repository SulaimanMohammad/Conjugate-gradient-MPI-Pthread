#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <libgen.h>



/*
 * Vecteur
 */
struct vector_s
{
    int N;          /* Vecteur de dimension N */
    double *elt;    /* elt[i] : i-ieme element du vecteur*/
};
typedef struct vector_s vector_t;


void vector_alloc(int N, vector_t *vec)
{
    vec->N = N;
    vec->elt = (double*)malloc(N*sizeof(double));
}

void vector_free(vector_t *vec)
{
    free(vec->elt);
}


/*
 * Initialise le vecteur à 0
 *   " vec = 0 "
 */
void vector_init_0(vector_t *vec)
{
    int i;

    for(i = 0 ; i < vec->N ; i++)
    {
	vec->elt[i] = 0.;
    }
}

/*
 * Multiplie tous les elements du vecteur par un scalaire
 *  " vec *= s "
 */
void vector_mul_scal(vector_t *vec, double s)
{
    int i;

    for(i = 0 ; i < vec->N ; i++)
    {
	vec->elt[i] *= s;
    }
}

/*
 * Affectation d'un vecteur par un autre multiplie' par un scalaire
 *  "  vec_out = s*vec_in  "
 */
void vector_affect_mul_scal(vector_t *vec_out, double s, vector_t *vec_in)
{
    int i;

    assert(vec_out->N == vec_in->N);

    for(i = 0 ; i < vec_in->N ; i++)
    {
	vec_out->elt[i] = s*vec_in->elt[i];
    }
}

/*
 * Calcule la norme L2 au carre' du vecteur
 *   "  (|| vec ||_2)²  "
 */
double vector_norm2(vector_t *vec)
{
    double norm2;
    int i;

    norm2 = 0.;

    for(i = 0 ; i < vec->N ; i++)
    {
	norm2 += vec->elt[i]*vec->elt[i];
    }

    return norm2;
}

/*
 * Additione à un vecteur un autre vecteur multiplie' par un scalaire
 *  " vec_inout += s*vec_in  "
 */
void vector_add_mul_scal(vector_t *vec_inout, double s, vector_t *vec_in)
{
    int i;

    assert(vec_inout->N == vec_in->N);

    for(i = 0 ; i < vec_in->N ; i++)
    {
	vec_inout->elt[i] += s*vec_in->elt[i];
    }
}

/*
 * Retourne le rapport de 2 produits scalaires
 *  "  (v1.w1) / (v2.w2)  "
 */
double div_bi_prod_scal(vector_t *v1, vector_t *w1, vector_t *v2, vector_t *w2)
{
    int i;

    assert(v1->N == w1->N);
    assert(v1->N == v2->N);
    assert(v1->N == w2->N);

    double scal1, scal2;

    scal1 = 0.;
    scal2 = 0.;

    for(i = 0 ; i < v1->N ; i++)
    {
	scal1 += v1->elt[i]*w1->elt[i];
	scal2 += v2->elt[i]*w2->elt[i];
    }

    return scal1/scal2;
}






/*
 *  Matrice creuse à 3 bandes
 *  (pour la ligne i, seules les colonnes i-1, i et i+1 sont non nulles)
 */
struct matrix3b_s
{
    int N;          /* Matrice de dimension NxN */

    /* Pour la ligne i,
     * A(i, i-1) = bnd[0][i]
     * A(i, i)   = bnd[1][i]
     * A(i, i+1) = bnd[2][i]
     * Tous les elements sur les colonnes autres que i-1, i et i+1 sont nuls
     */
    double *bnd[3];
};
typedef struct matrix3b_s matrix3b_t;

void linear_system_alloc_and_init(int N, matrix3b_t *A, vector_t *vb)
{
    assert(N > 2);
    int i;

    /* Allocations */
    A->N = N;
    A->bnd[0] = (double*)malloc(N*sizeof(double));
    A->bnd[1] = (double*)malloc(N*sizeof(double));
    A->bnd[2] = (double*)malloc(N*sizeof(double));

    vb->N   = N;
    vb->elt = (double*)malloc(N*sizeof(double));


    /* Remplissage */
    double coeff = 0.01;

    for(i = 0 ; i < N ; i++)
    {
	A->bnd[0][i] = -coeff;
	A->bnd[1][i] = 1. + 2*coeff;
	A->bnd[2][i] = -coeff;

	vb->elt[i] = 1.;
  }


    A->bnd[0][0] = 0.;
    vb->elt[0] = 1. + coeff;

    A->bnd[2][N-1] = 0.;
    vb->elt[N-1] = 1. + coeff;
}

void linear_system_free(matrix3b_t *A, vector_t *vb)
{
    free(A->bnd[0]);
    free(A->bnd[1]);
    free(A->bnd[2]);

    free(vb->elt);
}




/*
 * Produit Matrice-Vecteur
 *  " vy = A.vx  "
 */
void prod_mat_vec(vector_t *vy, matrix3b_t *A, vector_t *vx)
{
  assert(A->N == vx->N);
  assert(vy->N == vx->N);

  int i;

  /* cas i = 0 */
  i = 0;
  vy->elt[i] =
	A->bnd[1][i] * vx->elt[i] +
	A->bnd[2][i] * vx->elt[i+1];

  /* cas i = N-1 */
  i = A->N-1;
  vy->elt[i] =
	A->bnd[0][i] * vx->elt[i-1] +
	A->bnd[1][i] * vx->elt[i];

  /* Coeur de la matrice */
  for(i = 1 ; i < A->N-1 ; i++)
  {
    vy->elt[i] =
	  A->bnd[0][i] * vx->elt[i-1] +
	  A->bnd[1][i] * vx->elt[i] +
	  A->bnd[2][i] * vx->elt[i+1];
  }
}




/*
 * Algorithme du Gradient Conjugue'
 *   " Resoud le systeme A.vx = vb "
 */
void gradient_conjugue(matrix3b_t *A, vector_t *vb, vector_t *vx)
{
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

    /* Initialisation de l'algo */

    vector_init_0(vx);
    vector_affect_mul_scal(&vg, -1., vb);
    vector_affect_mul_scal(&vh, -1., &vg);
    sn = vector_norm2(&vg);

    /* Phase iterative de l'algo */

    for(k = 0 ; k < N && sn > seps ; k++)
    {
	printf("Iteration %5d, err = %.4e\n", k, sn);
	prod_mat_vec(&vw, A, &vh);

	sr = - div_bi_prod_scal(&vg, &vh, &vh, &vw);

	vector_add_mul_scal(vx, sr, &vh);
	vector_add_mul_scal(&vg, sr, &vw);

	sn1 = vector_norm2(&vg);

	sg = sn1 / sn;
	sn = sn1;

	vector_mul_scal(&vh, sg);

	vector_add_mul_scal(&vh, -1., &vg);
    }

    vector_free(&vg);
    vector_free(&vh);
    vector_free(&vw);
}


/* Verification du resultat
 *  A.vx "doit etre proche" de vb
 */
void verif_sol(matrix3b_t *A, vector_t *vb, vector_t *vx)
{
    vector_t vb_cal;
    double norm2;

    assert(A->N == vb->N);
    assert(A->N == vx->N);

    vector_alloc(A->N, &vb_cal);

    prod_mat_vec(&vb_cal, A, vx); /* vb_cal = A.vx */
    vector_add_mul_scal(&vb_cal, -1., vb); /* vb_cal = vb_cal - vb */
    norm2 = vector_norm2(&vb_cal);

    if (norm2 < 1.e-12)
    {
	printf("Resolution correcte du systeme\n");
    }
    else
    {
	printf("Resolution incorrecte du systeme, erreur : %.4e\n", norm2);
    }
}



/*
   Main
*/

int main(int argc, char **argv)
{
    int N;
    vector_t vx, vb;
    matrix3b_t A;

    if (argc != 2)
    {
	printf("Usage : %s <N>\n", basename(argv[0]));
	printf("\t<N>    : dimension de la matrice\n");
	abort();
    }
    N = atoi(argv[1]);


    /* Allocation et construction du systeme lineaire
       */
    linear_system_alloc_and_init(N, &A, &vb);
    vector_alloc(N, &vx); /* vx est l'inconnue */



    /* Resolution du systeme lineaire
     *  A.vx = vb
     * par application de l'algorithme de Gradient Conjugue'
     */
    gradient_conjugue(&A, &vb, &vx);


    /* Verification du resultat
     *  A.vx "doit etre proche" de vb
     */
    verif_sol(&A, &vb, &vx);



    /* Liberation memoire
     */
    linear_system_free(&A, &vb);
    vector_free(&vx);

    return 0;
}
