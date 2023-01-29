#
# Compile
#
main:	
	mpicc -pthread hyb_grad_conj.c -I ../mpi_decomp/ -I ../thr_decomp/ -I ../hyb_reduc/ -I ../hyb_exchg/ ../mpi_decomp/mpi_decomp.c ../thr_decomp/thr_decomp.c ../hyb_reduc/hyb_reduc.c ../hyb_exchg/hyb_exchg.c

clean:
	rm a.out
