/*
Neural Network for Classification
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h>

/* max set length */
#define SET_LEN 1000

/* performance/goal value -> stop cond */
#define GOAL 0.0001

/* max number of epochs for training */
#define MAX_EPOCHS 2000

/* network architecture */
#define INPUTS                 2
#define HIDDEN_NEURONS         5
#define OUT_NEURONS            1

/* precission for FP comparison */
#define EPSILON 0.000001

/* globals */

/*  eta - learning rate */
double eta = 0.01;

/* training data set */
struct training_set{
    /* x input val */
    double x_val;
    /* y input val */
    double y_val;
    /* corresponding output {+1, -1}*/
    int out;
};

/* testing data set */
struct testing_set{
    /* x input val */
    double x_val;
    /* y input val */
    double y_val;
};

/* randomization function for weight init */
float randomize()
{
    return (float)rand()/(float)RAND_MAX;
}

/* the tanh activation function for hidden layer */
double tanh_activation(double in){
     return (double)tanhf(in);
}

/* the linear activation function for output layer*/
int linear_activation(double in){
    return (in >= 0.0f) ? 1 : -1;
}

/* function that returns a set length */
int get_training_vector_length(struct training_set train_vector[SET_LEN]){
    int x;
    /* length to return */
    int len = 0;
    for(x=0;x<SET_LEN;x++){
        if(train_vector[x].x_val!=0.0f && train_vector[x].y_val!=0.0f && train_vector[x].out!=0) len++;
        else break;
    }
    return len;
}

int get_testing_vector_length(struct testing_set test_vector[SET_LEN]){
    int x;
    /* length to return */
    int len = 0;
    for(x=0;x<SET_LEN;x++){
        if(test_vector[x].x_val!=0.0f && test_vector[x].y_val!=0.0f) len++;
        else break;
    }
    return len;
}

/* get the content of the training and the test vectors */
void show_training_vector(struct training_set train_vector[SET_LEN], int len){
    int x;
    for(x=0;x<len;x++){
        printf("Train set input pair [%d] : %f | %f\n", x, train_vector[x].x_val, train_vector[x].y_val);
        printf("Train set output val [%d] : %d\n", x, train_vector[x].out);
    }
}


void show_testing_vector(struct testing_set test_vector[SET_LEN], int len){
    int x;
    for(x=0;x<len;x++){
        printf("Test set input pair [%d] : %f | %f\n", x, test_vector[x].x_val, test_vector[x].y_val);
    }

}

/* normalization function - in fact gets the max to define values domain mapping */
double* normalize(double x_buffer_train[SET_LEN], double y_buffer_train[SET_LEN], int len){
    /* vector to hold normalization values */
    double* norm = calloc(2, sizeof(double));
    /* indexes */
    int i;
    /* aux to store max */
    double max_x, max_y;
    /* init max-es */
    /* using abs to get the biggest value pos/neg and considering symmetry for values domain */
    max_x = fabs(x_buffer_train[0]);
    max_y = fabs(y_buffer_train[0]);
    /* loop */
    for(i=1;i<len;i++){
        if(fabs(x_buffer_train[i])>max_x) max_x = fabs(x_buffer_train[i]);
        if(fabs(y_buffer_train[i])>max_y) max_y = fabs(y_buffer_train[i]);
    }
    /* compose result */
    norm[0] = max_x;
    norm[1] = max_y;
    return norm;
}

/* entry point */
int main(int argc, char*argv[]){
    /* training values */
    double train_x_val, train_y_val;
    /* validation values */
    double test_x_val, test_y_val;
    /* neuron train output */
    int neuron_out;
    /* valid input values counter limited to 1000 */
    int input_idx=0;
    /* valid input values counter limited to 1000 for target values */
    int target_input_idx=0;
    /* training vector */
    struct training_set train_vector[SET_LEN];
    /* testing vector */
    struct testing_set test_vector[SET_LEN];
    /* input weights from input to hidden neurons */
    double input_weights[HIDDEN_NEURONS+1][INPUTS+1];
    /* delta input weights */
    double delta_input_weights[HIDDEN_NEURONS+1][INPUTS+1];
    /* output weight from hidden layer to output layer */
    double output_weights[OUT_NEURONS+1][HIDDEN_NEURONS+1];
    /* output weight delta */
    double delta_output_weights[OUT_NEURONS+1][HIDDEN_NEURONS+1];
    /* current number of training epochs */
    int epochs;
    /* train delta */
    int delta;
    /* sum of products in hidden layer */
    double input_sum[HIDDEN_NEURONS+1];
    /* delta input activation */
    double delta_input_activation[INPUTS+1];
    /* activation output for hidden layer */
    double hidden_activation[HIDDEN_NEURONS+1];
    /* activation output for hidden layer delta */
    double delta_hidden_activation[HIDDEN_NEURONS+1];
    /* the output sum of the neural network */
    double output_sum=0.0f;
    /* activation output for output layer */
    int output_activation;
    /* activation output for output layer delta */
    double delta_output_activation;
    /* training input index */
    int train_idx;
    /* bias */
    int bias=1;
    /* general purpose indexes */
    int i, j, h;
    /* additional storage for input values to be normalized */
    double x_buffer_train[SET_LEN], y_buffer_train[SET_LEN];
    /* normalization values */
    double* norm_train = calloc(2, sizeof(double));
        /* additional storage for test values to be normalized */
    double x_buffer_test[SET_LEN], y_buffer_test[SET_LEN];
    /* normalization values */
    double* norm_test = calloc(2, sizeof(double));
    /* global normalization value */
    double* norm = calloc(2, sizeof(double));
    /* global sse error */
    double sse_error, sse_error_old=0.0;
//    FILE *f = fopen("error9.mat","a+");

    /* loop and get training data */
    while(scanf("%lf,%lf,%d\n", &train_x_val, &train_y_val, &neuron_out)>0){
      /* if the input vector terminator wasn't received fill in training set */
      if(train_x_val!=0.0f && train_y_val!=0.0f && neuron_out!=0){
        /* populate the training set */
        train_vector[input_idx].x_val = train_x_val;
        train_vector[input_idx].y_val = train_y_val;
        train_vector[input_idx].out = neuron_out;
        /* additional storage for computing max - used in normalization sequence */
        x_buffer_train[input_idx] = train_x_val;
        y_buffer_train[input_idx] = train_y_val;
      }else break;
      /* check if max input reached */
      if(input_idx==1000) break;
      input_idx++;
    }

    /* get the normalization value for train set */
    norm_train = normalize(x_buffer_train, y_buffer_train, input_idx);

    /* reinit index */
    target_input_idx=0;
    /* the next input values are test/target values to validate against */
    while(scanf("%lf,%lf\n", &test_x_val, &test_y_val)>0){
            /* populate the testing set */
            test_vector[target_input_idx].x_val = test_x_val;
            test_vector[target_input_idx].y_val = test_y_val;
            /* check if max input reached */
      if(target_input_idx==1000) break;
      target_input_idx++;
    }

    /* get the normalization value for test set */
    norm_test=normalize(x_buffer_test, y_buffer_test, target_input_idx);

    /* get the unified normalization values */
    if(norm_test[0]>norm_train[0]) norm[0] = norm_test[0];
    else norm[0] = norm_train[0];
    if(norm_test[1]>norm_train[1]) norm[1] = norm_test[1];
    else norm[1] = norm_train[1];

    /* initialize weights and delta weights */

    /* initialization for input-hidden neurons */
    for(i=1;i<=HIDDEN_NEURONS;i++){
      for(h=0;h<=INPUTS;h++){
        input_weights[i][h]=randomize();
        delta_input_weights[i][h]=0.0f;
      }
    }

    /* initialization for hidden-output neurons */
    for(j=0;j<OUT_NEURONS;j++){
      for(i=0;i<=HIDDEN_NEURONS;i++){
        output_weights[j][i]=randomize();
        delta_output_weights[j][i]=0.0f;
        }
    }

     /* input sum init */
     for(i=1;i<=HIDDEN_NEURONS;i++){
            input_sum[i]=0.0f;
     }
	
     /* initializations */
     delta=0.0f;
     sse_error = 0.0f;
	
    /* neural network training sequence using backpropagation */
    int Ep = MAX_EPOCHS;
	
    /* iterate through training epochs */
    for(epochs=0;epochs<Ep;epochs++){

    /* iterate through training values */
    for(train_idx=0;train_idx<get_training_vector_length(train_vector);train_idx++){

        /* ----- Forward propagation ----- */

        /* update sum of products - inputs in hidden layer */
        for(i=1;i<=HIDDEN_NEURONS;i++){
                input_sum[i] = input_weights[i][0]*bias+
                               input_weights[i][1]*train_vector[train_idx].x_val/norm[0]+
                               input_weights[i][2]*train_vector[train_idx].y_val/norm[1];

               /* compute input hidden layer activation */
               hidden_activation[i] = tanh_activation(input_sum[i]);
        }

        /* hidden to output sum */

        /* output sum update */
        for(j=0;j<OUT_NEURONS;j++){
            output_sum = output_weights[j][0]*bias;
            for(i=1;i<=HIDDEN_NEURONS;i++){
                output_sum += output_weights[j][i]*hidden_activation[i];
            }
        }

        /* compute output */
        output_activation = linear_activation(tanh_activation(output_sum));

        /* update the delta */
        delta = train_vector[train_idx].out-output_activation;

        /* error update */
        sse_error+=0.5*pow(delta,2);

//      fprintf(f, "%f,%d\n", sse_error, epochs);

        /* delta output activation */
        delta_output_activation = delta*(1.0-pow(tanh_activation(output_sum),2));

        /* ----- Error backpropagation ----- */

         /* initialize delta hidden activation */

        /* update delta hidden output */
        for(j=0;j<OUT_NEURONS;j++){
         for(i=1;i<=HIDDEN_NEURONS;i++){
            delta_hidden_activation[i] = delta_output_activation*
                                         output_weights[j][i]*
                                         (double)(1.0-pow(hidden_activation[i],2));
         }
       }

        /* delta input */
        for(h=1;h<=INPUTS;h++){
            for(i=1;i<=HIDDEN_NEURONS;i++){
                delta_input_activation[h] += input_weights[i][h]*
					     delta_hidden_activation[i];
            }
        }

        /* output weights delta */
        for(j=0;j<OUT_NEURONS;j++){
            delta_output_weights[j][0] = eta*
                                         delta_output_activation;
            output_weights[j][0]+=delta_output_weights[j][0];

          for(i=1;i<=HIDDEN_NEURONS;i++){
            delta_output_weights[j][i] = eta*
                                         delta_output_activation*
                                         hidden_activation[i];
            output_weights[j][i]+=delta_output_weights[j][i];
          }
        }

        /* weights delta computation and weights update */
           for(i=1;i<=HIDDEN_NEURONS;i++){
            /* input weights deltas */
            delta_input_weights[i][0] = eta*
                                        bias*
                                       delta_hidden_activation[i];
             input_weights[i][0]+=delta_input_weights[i][0];

             delta_input_weights[i][1] = eta*
                                         train_vector[train_idx].x_val/norm[0]*
                                         delta_hidden_activation[i];
             input_weights[i][1]+=delta_input_weights[i][1];

            delta_input_weights[i][2] = eta*
                                         train_vector[train_idx].y_val/norm[1]*
                                         delta_hidden_activation[i];
             input_weights[i][2]+=delta_input_weights[i][2];

        }
      }
        /* update training error */
        sse_error_old = sse_error;

        /* stop condition - if performance goal reached */
        if(sse_error==GOAL) break;
  }

    /* test the network */
    /* iterate through test values */
    int t=0;
    for(t=0;t<get_testing_vector_length(test_vector);t++){

        /* update sum of products - inputs in hidden layer */
        for(i=1;i<=HIDDEN_NEURONS;i++){
                input_sum[i] = input_weights[i][0]*bias+
                               input_weights[i][1]*test_vector[t].x_val/norm[0]+
                               input_weights[i][2]*test_vector[t].y_val/norm[1];
		/* compute input hidden layer activation */
                hidden_activation[i] = tanh_activation(input_sum[i]);
        }

	/* compute output sum with the computed weights */
        for(j=0;j<OUT_NEURONS;j++){
            output_sum = output_weights[j][0]*bias;
            for(i=1;i<=HIDDEN_NEURONS;i++){
                output_sum += output_weights[j][i]*hidden_activation[i];
            }
        }

	output_activation = linear_activation(tanh_activation(output_sum));

        /* print results for test set */
        if(output_activation==1) printf("+%d\n",output_activation);
        else printf("%d\n",output_activation);
    }
    if(norm_train) free(norm_train);
    if(norm_test) free(norm_test);
    if(norm) free(norm);
//    fclose(f);
    return 0;
}

