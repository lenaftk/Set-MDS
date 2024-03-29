#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "mds_pertubations.h"

#define ERROR_BUF_SZ 2000
#define DBL_MAX 1.79769e+308
double error_buf[ERROR_BUF_SZ] = {DBL_MAX};

/*
 * Max number of dimensions to reduce to is 1000
 */

double
single_pertub_error(double* d_current, double* d_goal,
                    double* xs, int row, int pertub_dim,
                    int x_rows, int x_cols, double step)
{
    double error = 0;
    int ll;
    int d_idx = x_rows * row;
    int x_idx = x_cols * row;

    //#pragma omp parallel for reduction (+:error)
    for(ll = 0; ll < x_rows; ll++)
    {
        double d_prev, before, after, diff1, diff;
        if(row != ll)
        {
            d_prev = d_current[d_idx + ll] * d_current[d_idx + ll];
            diff1 = (xs[x_idx + pertub_dim] - xs[ll * x_cols + pertub_dim]); //#afairesi to simeio pou kounietai (i diastasi tou) - ola ta ipoloipa simeia stin sugkekrimeni diastasi   
            before = diff1 * diff1;  //vriskw tin diafora sto sugkekrimeni diastasi
            after = (diff1 + step) * (diff1 + step);  // briskw tin diafora sto step
            diff = d_goal[d_idx + ll] - sqrt(d_prev - before + after);
            error += diff * diff;
        }
    }
    return error;
}



pertub_res
min_pertub_error(double* xs, double radius, double* d_current,
                    double* d_goal, int ii, int x_rows, int x_cols,double prop_thr, double prop_step, double* prop_matrix, int turn,
                    double percent, int n_jobs)
{
    int jj;
    struct pertub_res optimum;
    optimum.error = DBL_MAX;
    optimum.k = 0;
    int prop_idx = 2 * x_cols * ii;
    int min_prop_k = 0;

#pragma omp parallel num_threads(n_jobs)
    {
         int time_ = (int)time(NULL) ^ omp_get_thread_num() ^ ii  ^ turn ;
         srand(time_);
#pragma omp for nowait
        for(jj=0; jj < 2 * x_cols; jj++)
        {
            double random_number = (double)rand() / (double)((unsigned)RAND_MAX);
            if (random_number > prop_matrix[prop_idx + jj]){
                error_buf[jj] = DBL_MAX;
                continue;
            }
            double step = jj < x_cols ? radius : -radius;
            error_buf[jj] = single_pertub_error(
                d_current, d_goal, xs, ii, jj % x_cols,
                x_rows, x_cols, step);
        }
    }

    for(jj=0; jj<2*x_cols; jj++){
        if(error_buf[jj] < DBL_MAX && prop_matrix[prop_idx + jj] > prop_thr){
            prop_matrix[prop_idx + jj] -= prop_step;
        }
    }
    for(jj=0; jj < 2 * x_cols; jj++) {
        if(error_buf[jj] < optimum.error) {
            optimum.k = jj % x_cols;
            optimum.step = jj < x_cols ? radius : -radius;
            optimum.error = error_buf[jj];
            min_prop_k = jj;
        }
    }
    if(prop_matrix[prop_idx + min_prop_k] + 2*prop_step <= 1.00){
        prop_matrix[prop_idx + min_prop_k] += 2*prop_step;
    }
    
    //print propability matrix
    // for(jj=0; jj<2*x_cols; jj++){
    //      printf(" %f ",prop_matrix[prop_idx + jj]);
    // }
    // printf("\n");

    return optimum;
}