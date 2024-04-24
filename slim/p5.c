/*
Neural Network for Regression for Noisy Data
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <float.h>

/* max set length */
#define SET_LEN 1000
#define LARGE_SET 750

/* max number of epochs for training and error thresholds */
#define MAX_EPOCHS 100000
#define TRAIN_ERR_H 0.995
#define TRAIN_ERR_L 1.005

/* network architecture */
#define INPUTS                 2 // added the bias
#define HIDDEN_LAYERS	       2          
#define HIDDEN_NEURONS         4
#define OUT_NEURONS            1   

#define BIAS -1
#define MAX_NEURONS 11

/* utils for fp */
#define FP_TOL 0.0001f

/* learning rate */
double eta=0.03f;

/* training data set */
struct training_set{
    /* x input val */
    double x_val;
    /* corresponding output */
    double out;
};

/* testing data set */
struct testing_set{
    /* x input val */
    double x_val;
};

/* the tanh_activation activation function for hidden layer */
double tanh_activation(double in){
    return (double)tanhf(in);
}

/* the linear activation function for output layer*/
double linear_activation(double in){
    return in;
}

/* function that returns a set length */
int get_training_vector_length(struct training_set train_vector[SET_LEN]){
    int x;
    /* length to return */
    int len = 0;
    for(x=0;x<SET_LEN;x++){
        if(train_vector[x].x_val!=0.0f && train_vector[x].out!=0.0f)len++;
        else break;
    }
    return len;
}

int get_testing_vector_length(struct testing_set test_vector[SET_LEN]){
    int x;
    /* length to return */
    int len = 0;
    for(x=0;x<SET_LEN;x++){
        if(test_vector[x].x_val!=0.0f) len++;
        else break;
    }
    return len;
}

/* randomization function for weight init */
double randomize(){
    /* number init */
    int num_in = 0;
    /* make it double */
    double randomize_in;
    /* test number value */
    while (num_in == 0){
        num_in = (rand() % 100) + 1;
    }
    /* problem specific */
    if ((num_in % 2) == 0) num_in = -num_in;
    randomize_in = num_in;
    /* apply thresholds */ 
    if ((randomize_in > 10.0f) || (randomize_in < -10.0f))
        return (randomize_in/= 100.0f);
    return (randomize_in/=10.0f);
}

/* test and update max value */
double get_max(double cur_val, double cur_max){
	if(fabs(cur_val)-cur_max>FP_TOL) return fabs(cur_val);
	return cur_max;
}

/* normalize a given value in the nn input data set */
double normalize(double val, double norm_factor, double max_val){
	return (val-norm_factor)/max_val;
}

/* entry point */
int main()
{
    /* training vector */
    struct training_set train_vector[SET_LEN];
    /* testing vector */
    struct testing_set test_vector[SET_LEN];
    /* flag for desired error value reach */
    short goal_reached=0;
    /* indexes for train and test data sets */
    int input_index=0, test_input_index=0;
    /* general purpose indexes */
    int t=0, i, h, j;
    /* number of training epochs */
    int epochs=0;
    /* train and test data sets ; train data sets with 2 values for in and out */
    double data_train [SET_LEN][2], data_test [SET_LEN];
    /* neuron in the network */
    /* first coordinate : layer
       second coordinate : neuron index in layer
       third coordinate : out / delta / net out / weights */
    double neuron_unit[MAX_NEURONS][MAX_NEURONS][MAX_NEURONS]; 
    /* network output */
    double net_out=1.0f;
    /* normalization useful values */
    double norm_val_in=0.0f, max_val_in=0.0f;
    double norm_val_out=0.0f, max_val_out=0.0f;

    /* get train data */
    while(scanf("%lf,%lf\n", &data_train[t][0], &data_train[t][1])>0)
    {
        if ((data_train[t][0] == 0.0f) && (data_train[t][1] == 0.0f))
        {
            input_index = t;
            break;
        }
	/* populate the training set */
        train_vector[t].x_val = data_train[t][0];
        train_vector[t].out = data_train[t][1];
	/* compute normalization factor */
	norm_val_in += data_train[t][0];
        norm_val_out += data_train[t][1];        
 	/* get max */
	max_val_in = get_max(data_train[t][0], max_val_in);
	max_val_out = get_max(data_train[t][1], max_val_out);      
      /* check max length */
      if(t==SET_LEN-1) break;
      t++;
    }
    /* update normalization factor */
    norm_val_out /= input_index;
    norm_val_in /= input_index;
    
    /* get test data set */
    for (t = 0; t < SET_LEN; t++)
    {
        if (scanf("%lf", &data_test[t]) <= 0)
        {
	    /* number of input values for test */
            test_input_index = t;
            break;
        }
        /* populate the testing set */
        test_vector[t].x_val = data_test[t];
    }
    
    /* train data normalization */ 
    for (t = 0; t < input_index; t++)
    {
        data_train[t][0] = normalize(data_train[t][0], norm_val_in, max_val_in);
	data_train[t][1] = normalize(data_train[t][1], norm_val_out, max_val_out);
    }
    
    /* test data normalization */ 
    for (t = 0; t < (test_input_index); t++)
    {
        data_test[t] = (data_test[t] - norm_val_in) / max_val_in;
    }
    
    /* init weights */
	
    /* input layer init */
    for (i = 0,h=0; i < INPUTS; i++)
    {
        neuron_unit[h][i][3] = randomize();
    }
        
    /* first hidden layer init */    
    for (i = 0, h=1; i < HIDDEN_NEURONS; i++){
        for (j = 0; j < INPUTS + 1; j++){
            neuron_unit[h][i][j+3] = randomize();
        }
    }
    
    /* second hidden layer init */
    for (i = 0, h=2; i < HIDDEN_NEURONS; i++){
        for (j = 0; j < HIDDEN_NEURONS+1; j++){
            neuron_unit[h][i][j+3] = randomize();
        }
    }
    
    /* output layer init */
    for (j = 0, h = HIDDEN_LAYERS + 1, i = 0; j < HIDDEN_NEURONS+1; j++){
        neuron_unit[h][i][j+3] = randomize();
    }

    /* adjust learning rate according set size */
    if (input_index > LARGE_SET) eta = 0.01;

    /* loop for training the net */
    while (1)
    {   /* check if reached the max number of epochs */
        if (epochs > MAX_EPOCHS) break;
        /* stop if goal was reached */
        if(goal_reached==1) break;        
        /* loop through input patterns */
        for (t = 0; t < input_index; t++)
        {
            /* forward propagation */

	    /* input activation */
            for (i = 0, h = 0; i < INPUTS; i++){
                neuron_unit[h][i][0] = neuron_unit[h][i][3] * data_train[t][0];
                neuron_unit[h][i][2] = tanh_activation(neuron_unit[h][i][0]);
            }
            /* first hidden activation */
            for (i = 0, h = 1; i < HIDDEN_NEURONS; i++){
                neuron_unit[h][i][0] = 0;
                for (j = 0; j < INPUTS; j++){
                   neuron_unit[h][i][0] = neuron_unit[h][i][0] + neuron_unit[h][i][j+3] * neuron_unit[h-1][j][2];
                }
                neuron_unit[h][i][0] = neuron_unit[h][i][0] + neuron_unit[h][i][j+4] * BIAS;
                neuron_unit[h][i][2] = tanh_activation(neuron_unit[h][i][0]);
            }
                
            /* second hidden layer activation */
            for (i = 0, h = 2; i < HIDDEN_NEURONS; i++){
                neuron_unit[h][i][0] = 0;
                for (j = 0; j < HIDDEN_NEURONS; j++){
                    neuron_unit[h][i][0] = neuron_unit[h][i][0] + neuron_unit[h][i][j+3] * neuron_unit[h-1][j][2];
                }
                neuron_unit[h][i][0] = neuron_unit[h][i][0] + neuron_unit[h][i][j+4] * BIAS;
                neuron_unit[h][i][2] = tanh_activation(neuron_unit[h][i][0]);
            }
            
            /* output layer */ 
            for (i = 0, h = HIDDEN_LAYERS + 1, neuron_unit[h][0][0] = 0; i < HIDDEN_NEURONS; i++){
                neuron_unit[h][0][0] = neuron_unit[h][0][0] + neuron_unit[h][0][i+3] * neuron_unit[h-1][i][2];
            }
            neuron_unit[h][0][0] = neuron_unit[h][0][0] + neuron_unit[h][0][i+4] * BIAS;
            neuron_unit[h][0][2] = (neuron_unit[h][0][0]);
            net_out = linear_activation(neuron_unit[h][0][2]);
                
            /* backpropagation */
            
            /* check the trainig error and update */
            if (((net_out/data_train[t][1]) < TRAIN_ERR_H) || 
		((net_out/data_train[t][1]) > TRAIN_ERR_L)){
                /* update deltas */

                /* for output layer */
                neuron_unit[HIDDEN_LAYERS+1][0][1] = (data_train[t][1] - net_out);
                
                /* second hidden layer */               
                for (i = 0, h = HIDDEN_LAYERS; i < HIDDEN_NEURONS; i++){
                    neuron_unit[h][i][1] = neuron_unit[h+1][0][1] * neuron_unit[h+1][0][i+3] * (1 - (neuron_unit[h][i][2] * neuron_unit[h][i][2]));
                }
                
                /* first hidden layer */    
                for (i = 0, h = 1; i < HIDDEN_NEURONS; i++){
                    neuron_unit[h][i][1] = 0;
                    for (j = 0; j < HIDDEN_NEURONS; j++)
                    {
                        neuron_unit[h][i][1] = neuron_unit[h][i][1] + neuron_unit[h+1][j][1] * neuron_unit[h+1][j][i+3] * (1 - (neuron_unit[h][i][2] * neuron_unit[h][i][2]));
                    }
                }
                
		/* input layer */
                for (i = 0, h = 0; i < INPUTS; i++){
                    neuron_unit[h][i][1] = 0;
                    for (j = 0; j < HIDDEN_NEURONS; j++)
                    {
                        neuron_unit[h][i][1] = neuron_unit[h][i][1] + neuron_unit[h+1][j][1] * neuron_unit[h+1][j][i+3] * (1 - (neuron_unit[h][i][2] * neuron_unit[h][i][2]));
                    }
                }
                    
        
                /* weights update */
                
                /* input layer */
                for (i = 0, h = 0; i < INPUTS; i++){
                    neuron_unit[h][i][3] = neuron_unit[h][i][3] + eta * neuron_unit[h][i][1] * data_train[t][0];
                }
                
                /* first hidden layer */
                for (i = 0, h = 1; i < HIDDEN_NEURONS; i++){
                    for (j = 0; j < INPUTS; j++){
                        neuron_unit[h][i][j+3] = neuron_unit[h][i][j+3] + eta * neuron_unit[h][i][1] * neuron_unit[h-1][j][2];
                    }
                    neuron_unit[h][i][j+4] = neuron_unit[h][i][j+4] + eta * neuron_unit[h][i][1] * BIAS;
                }
                    
                /* second hidden layer */
                for (i = 0, h = 2; i < HIDDEN_NEURONS; i++){
                    for (j = 0; j < HIDDEN_NEURONS; j++){
                        neuron_unit[h][i][j+3] = neuron_unit[h][i][j+3] + eta * neuron_unit[h][i][1] * neuron_unit[h-1][j][2];
                    }
                    neuron_unit[h][i][j+4] = neuron_unit[h][i][j+4] + eta * neuron_unit[h][i][1] * BIAS;
                }
                
                /* output layer */
                for (j = 0, i = 0, h = HIDDEN_LAYERS + 1; j < HIDDEN_NEURONS; j++){
                    neuron_unit[h][i][j+3] = neuron_unit[h][i][j+3] + eta * neuron_unit[h][i][1] * neuron_unit[h-1][j][2];
                }
                neuron_unit[h][i][j+4] = neuron_unit[h][i][j+4] + eta * neuron_unit[h][i][1] * BIAS;

            }
	    else goal_reached=0;
        } 
	epochs++;
    } 
    
    /* test the trained network */
    for (t = 0; t < (test_input_index); t++)
    {
        
       /* forward propagation with test data */

	    /* input activation */
            for (i = 0, h = 0; i < INPUTS; i++){
                neuron_unit[h][i][0] = neuron_unit[h][i][3] * data_test[t];
                neuron_unit[h][i][2] = tanh_activation(neuron_unit[h][i][0]);
            }
            /* first hidden activation */
            for (i = 0, h = 1; i < HIDDEN_NEURONS; i++){
                neuron_unit[h][i][0] = 0;
                for (j = 0; j < INPUTS; j++){
                   neuron_unit[h][i][0] = neuron_unit[h][i][0] + neuron_unit[h][i][j+3] * neuron_unit[h-1][j][2];
                }
                neuron_unit[h][i][0] = neuron_unit[h][i][0] + neuron_unit[h][i][j+4] * BIAS;
                neuron_unit[h][i][2] = tanh_activation(neuron_unit[h][i][0]);
            }
                
            /* second hidden layer activation */
            for (i = 0, h = 2; i < HIDDEN_NEURONS; i++){
                neuron_unit[h][i][0] = 0;
                for (j = 0; j < HIDDEN_NEURONS; j++)
                {
                    neuron_unit[h][i][0] = neuron_unit[h][i][0] + neuron_unit[h][i][j+3] * neuron_unit[h-1][j][2];
                }
                neuron_unit[h][i][0] = neuron_unit[h][i][0] + neuron_unit[h][i][j+4] * BIAS;
                neuron_unit[h][i][2] = tanh_activation(neuron_unit[h][i][0]);
            }
            
            /* output layer */ 
            for (i = 0, h = HIDDEN_LAYERS + 1, neuron_unit[h][0][0] = 0; i < HIDDEN_NEURONS; i++){
                neuron_unit[h][0][0] = neuron_unit[h][0][0] + neuron_unit[h][0][i+3] * neuron_unit[h-1][i][2];
            }
            neuron_unit[h][0][0] = neuron_unit[h][0][0] + neuron_unit[h][0][i+4] * BIAS;
            neuron_unit[h][0][2] = (neuron_unit[h][0][0]);
            net_out = linear_activation(neuron_unit[h][0][2]);
	    /* de-normalize output */
            net_out = neuron_unit[h][0][2] * max_val_out + norm_val_out;
        
         /* output */
        printf("%lf\n", net_out);
        
    }
    
    return 0;
}


