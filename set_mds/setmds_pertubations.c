#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "setmds_pertubations.h"

#define ERROR_BUF_SZ 2000
#define DBL_MAX 1.79769e+308
double error_buf[ERROR_BUF_SZ] = {DBL_MAX};

/*
 * Max number of dimensions to reduce to is 1000
 */

void enter(){
    printf("\n");
}

double
single_pertub_error(int n_sets, long *sets, double* d_point_temp, double* d_current_sets, double* d_current, double* d_goal,
                    double* xs, int row, int pertub_dim,
                    int x_rows, int x_cols, double step)
{
    double error = 0, diff;
    int ll,jj;
    int d_idx = x_rows * row;
    int x_idx = x_cols * row;
    int d_goal_idx = n_sets * sets[row];
   // printf("arxi \n");
   // enter();

    //#pragma omp parallel for reduction (+:error)
    for(ll = 0; ll < x_rows; ll++)
    {
       // printf("point= %d , ll= %d ", row, ll);
        double d_prev, before, after, diff1,  d_current_temp;
        if(row != ll)
        {
            d_prev = d_current[d_idx + ll] * d_current[d_idx + ll];
            diff1 = (xs[x_idx + pertub_dim] - xs[ll * x_cols + pertub_dim]);
            before = diff1 * diff1;
            after = (diff1 + step) * (diff1 + step);
            d_current_temp = sqrt(d_prev - before + after);
            if (d_current_temp < d_point_temp[sets[ll]]){
                d_point_temp[sets[ll]] = d_current_temp;
            }
        }
    }
   // enter();
        for(jj = 0; jj < n_sets; jj++){
            diff = d_goal[d_goal_idx + sets[jj]] - d_point_temp[jj];
           // printf(" error %f -  ", error);
            error += diff*diff;
            
        }
    //    enter();
    return error;
}



pertub_res
min_pertub_error( int n_sets, long *sets, double* d_current_sets, double* xs, double radius, double* d_current,
                    double* d_goal, int ii, int x_rows, int x_cols, int turn,
                    double percent, int n_jobs)
{
    int jj,hh;
    struct pertub_res optimum;
    optimum.error = DBL_MAX;
    optimum.k = 0;
    int prop_idx = 2 * x_cols * ii;
    int min_prop_k = 0;
    double d_point_temp[n_sets];


#pragma omp parallel num_threads(n_jobs)
    {
         int time_ = (int)time(NULL) ^ omp_get_thread_num() ^ ii  ^ turn ;
         srand(time_);
#pragma omp for nowait
        for(jj=0; jj < 2 * x_cols; jj++)
        {
            //double random_number = (double)rand() / (double)((unsigned)RAND_MAX);
            // if (random_number > prop_matrix[prop_idx + jj]){
            //     error_buf[jj] = DBL_MAX;
            //     continue;
            // }
            for (hh=0; hh<n_sets; hh++){
                d_point_temp[hh] = d_current_sets[n_sets*sets[ii] + hh];
            }
            double step = jj < x_cols ? radius : -radius;
            error_buf[jj] = single_pertub_error(n_sets, sets, d_point_temp, d_current_sets,
                d_current, d_goal, xs, ii, jj % x_cols,
                x_rows, x_cols, step);
        }
    }

    // for(jj=0; jj<2*x_cols; jj++){
    //     if(error_buf[jj] < DBL_MAX && prop_matrix[prop_idx + jj] > prop_thr){
    //         prop_matrix[prop_idx + jj] -= prop_step;
    //     }
    // }
    for(jj=0; jj < 2 * x_cols; jj++) {
        if(error_buf[jj] < optimum.error) {
            optimum.k = jj % x_cols;
            optimum.step = jj < x_cols ? radius : -radius;
            optimum.error = error_buf[jj];
            min_prop_k = jj;
        }
    }
    // if(prop_matrix[prop_idx + min_prop_k] + 2*prop_step <= 1.00){
    //     prop_matrix[prop_idx + min_prop_k] += 2*prop_step;
    // }
    

    return optimum;
}