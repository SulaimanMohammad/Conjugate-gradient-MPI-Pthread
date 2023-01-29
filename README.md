# Conjugate-gradient-MPI-Pthread
Parallelize algorithm of conjugate gradient MPI-Pthread

# Hyper_reduce

**1)shared_reduc_init** : used to initialise semaphores, mutexes, barrier needed in the function with
malloc for each one

red_mutex: is used to protect the data in the function because there are many threads and mutex is used
to avoid Race condition.
red_barrier: used to synchronise the threads and wait all the threads to finish part of the code to
continue the rest so the value in the barrier is equal to the number of threads.
Semph: used to acess part of the code by just on thread so it is binary semaphore.
semph_th_num: this semaphore is used to chzck that all the threads executed the instructions , so the
value is the number of threads and to check if all the threads executed the function we use
sem_getvalue and if it is zeo that means all the threads called”wait” before

**2)shared_reduc_destroy:** used to free the dynamic memory allocated by malloc and to destroy
semaphores, mutexes and barrier.

**3)hyb_reduc_sum** : the function where we compute “out “

- at the top of the instruction the semph_th_num is used to count the number of threads coputed this
code and in the end I used sem_getvalue to rest the thread master
-insid a mutex values of “in “ for nvals are accumulated in red_val and barrier used until all the threads
executed in parralle to finish
- chose a thread master whih is the first one since there are multiple process we use MPI_Gather from
all and the root is the process 0. when all that done in the root we sum the data to have the “out”
- we need to distribute all the data after to all the processes and save them in red_val
- we will do same steps for the threads not master “out” is red_val
-MPI_Barrier to synchro all the processes

*the result of executing the test was right


# Hyper_exchg

**1) shared_exchg_init**

- **semph** : it is semaphore to do one thread by time and it is init as a semaphore shared between the
processes
- **exc_mutex** : used too to protect data
- initi value **left_s, right_s** as zeors and they are value in exchange

**2) shared_exchg_destroy**
to destroy the mutex and the semaphore

**3) hyb_exchg**

- we chose master threads and as it is in if condition with **chose_th_master** as zero then the first thread
acess the function will be the master
- if we have just one process then nothing to exchange and the value to be in exchange will stay as they
initialized.
-if the master thread is from the first process then nothing to exchange in the left and we recive the
right , and the process send **sh_arr[mpi_decomp->mpi_nloc – 1]** , blocking synchronous send is used
because it is better for multithreads
-if the master thread is from the last process then nothing to exchange in the right and we recive the left
, and the process **ssh_arr[0]** , blocking synchronous send is used because it is better for multithreads
-if threads is in between the first and the last one then we need to send and receive the values
-assign **left_s, right_s** to the pointer of output
- for the rest of threads **left_s, right_s** are shared so we can assign immediately and here mutex is
used

*the results of executing the test was almost good mpirun -n H ./a.out K , K>H

# hyper_grad_conj.c

it is the implementation of the previous codes by using the sequential code.

-In the main , initi the MPI multi thread , the Structur, **shared_exchg_init** , **shared_reduc_init** and init
the system and the vector

- the algorithm is in **gradient_conjugue** so because we divide the interval and we have many threads
then we need to execute the algorithm for all threads, so gradient_conjugue is the routine and it takes
arg from the main structur.
- exactly same for **verif_sol.**


important points in the code

1- in function “ **vector_norm2** ” it should be done on the vector in all the threads so we need to use
“ **hyb_reduc_sum** “ here to have one result.

2- in the function “ **div_bi_prod_scal** ” this operation should be done for all to reurn on global value so
“ **hyb_reduc_sum** ” is used here too

3- in function “ **prod_mat_vec** ” here the rank of the process should be disscussed and the rank of the
thread too, and we use **hyb_exchg**
* first proccess

- first thread nothing to exchange
- last thread we need to have val_right from hyb_reduc_sum
-for threads between nothing to exchange

*last thread

- first thread we need to have val_left hyb_reduc_sum
- last thread and in between nothing to exchange

* processes not first or last one

- first thread we need the val_left
-last thread val_right
-threads between nothing to exchange.

And we wait the threads and the MPI to continue

Note: I did not have time to debug and check ( a problm with N in linear_system_alloc_and_init) or
change the idea of **hyper_grad_conj** because I sarted so late, and couldn’t translate the report to
french.



