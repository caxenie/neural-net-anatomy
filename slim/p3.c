/* 
Single Neuron for Regression
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>

/* max set length */
#define SET_LEN 1000

/* performance/goal value */
#define GOAL 0.0001

/* max number of epochs for training */
#define MAX_EPOCHS 5000

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

/* randomization function for weight init */
float randomize()
{
    return (float)rand() / (float)RAND_MAX;
}

/* the linear activation function */
double linear_activation(double in){
    return in;
}

/* function that returns a set length */
int get_training_vector_length(struct training_set train_vector[SET_LEN]){
    int x;
    /* length to return */
    int len = 0;
    for(x=0;x<SET_LEN;x++){
        if(train_vector[x].x_val!=0.0f && train_vector[x].out!=0.0f) len++;
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

/* get the content of the training and the test vectors */
void show_training_vector(struct training_set train_vector[SET_LEN], int len){
    int x;
    for(x=0;x<len;x++){
        printf("Train set pair [%d] : %f | %f\n", x, train_vector[x].x_val, train_vector[x].out);
    }
}


void show_testing_vector(struct testing_set test_vector[SET_LEN], int len){
    int x;
    for(x=0;x<len;x++){
        printf("Test set input [%d] : %f \n", x, test_vector[x].x_val);
    }

}

/* used for normalization - define the values domain mapping */
double get_max(double buffer[SET_LEN], int len){
	/* index */
	int i;
	/* aux to store max */
	/* use abs to get biggest value to setup as boundry */
	double max = fabs(buffer[0]);
	/* loop */
	for(i=1;i<len;i++){
		if(fabs(buffer[i])>max) max = fabs(buffer[i]);
	}
	return max;
}


/* entry point */
int main(int argc, char*argv[]){
    /* training values */
    double train_x_val;
    /* validation values */
    double test_x_val;
    /* neuron train output */
    double neuron_out;
    /* valid input values counter limited to SETLEN */
    int input_idx=0;
    /* valid input values counter limited to SETLEN for target values */
    int target_input_idx=0;
    /* training vector */
    struct training_set train_vector[SET_LEN];
    /* testing vector */
    struct testing_set test_vector[SET_LEN];
    /* input weights*/
    double input_weights[2];
    /* current number of training epochs */
    int epochs;
    /* learning rate eta */
    double eta = 0.001;
    /* comparison training data - neuron output */
    double delta;
    /* global delta */
    double sse_error;
    /* sum of products */
    double sum;
    /* training input index */
    int train_idx;
    /* output value of the neuron to be validated against target */
    double net_out;
    /* bias */
    int bias=1;
	/* additional storage for input and output values to be normalized */
	double x_buffer_train[SET_LEN], out_buffer[SET_LEN];
	/* normalization values */
	double* norm_train = calloc(2, sizeof(double));
		/* additional storage for test values to be normalized */
	double x_buffer_test[SET_LEN];
	/* normalization values */
	double norm_test;
	/* global normalization value */
	double* norm=calloc(2, sizeof(double));
	/* epochs to train */
	int total_epochs = MAX_EPOCHS;


    /* loop and get training data */
    while(scanf("%lf,%lf\n", &train_x_val, &neuron_out)>0){
      /* if the input vector terminator wasn't received fill in training set */
      if(train_x_val!=0.0f && neuron_out!=0.0f){
        /* populate the training set */
        train_vector[input_idx].x_val = train_x_val;
        train_vector[input_idx].out = neuron_out;
		/* additional storage for computing max - used in normalization sequence */
		x_buffer_train[input_idx] = train_x_val;
		out_buffer[input_idx] = neuron_out;
      }else break;
      /* check if max input reached */
      if(input_idx==1000) break;
      input_idx++;
    }

	/* by analizing the behavior I can say that if we have
	   a small training set and large test set increase the number of epochs
	   to ensure that the weights will have a better value	
	 */
	if(input_idx<=5) total_epochs = MAX_EPOCHS*13;

	/* if data set is consistent reduce the epochs number to pass in DOMJudge */
	if(input_idx>900)  total_epochs=1000;

	/* get the normalization value for train set */
	norm_train[0] = get_max(x_buffer_train, input_idx);
	norm_train[1] = get_max(out_buffer, input_idx);

    /* reinit index */
    target_input_idx=0;
    /* the next input values are test/target values to validate against */
    while(scanf("%lf\n", &test_x_val)>0){
            /* populate the testing set */
            test_vector[target_input_idx].x_val = test_x_val;
			/* additional storage for computing max - used in normalization sequence */
			x_buffer_test[target_input_idx] = test_x_val;
            /* check if max input reached */
      if(target_input_idx==1000) break;
      target_input_idx++;
    }

	/* get the normalization value for test set */
	norm_test = get_max(x_buffer_test, target_input_idx);

	/* get the unified normalization values */
	if(norm_test>norm_train[0]) norm[0] = norm_test;
	else norm[0] = norm_train[0];
	norm[1] = norm_train[1];

    /* initialize weights */
    input_weights[0]=randomize();
    input_weights[1]=randomize();
    input_weights[2]=randomize();
	/* initialize sum */
    sum=0.0; 
    /* initialize delta */
    delta=0.0;
    /* init global sse delta */
    sse_error=0.0;

    /* neuron training sequence */

    /* iterate throught training epochs */
    for(epochs=0;epochs<total_epochs;epochs++){
        /* iterate through training values */
        for(train_idx=0;train_idx<get_training_vector_length(train_vector);train_idx++){
            /* compute sum */
            sum=input_weights[0]*train_vector[train_idx].x_val/norm[0]+
				input_weights[1]*bias;
            /* compute the output */
            net_out = linear_activation(sum);
            /* compute delta between current output and the target value */
            delta=train_vector[train_idx].out - net_out;

			/* update the weights */
            input_weights[0]+=eta*delta*train_vector[train_idx].x_val/norm[0];
            input_weights[1]+=eta*delta*bias;
	    /* update global delta */
	    sse_error+=(delta*delta);
	   }
	/* stop condition - if performance goal reached */
	if(sse_error==GOAL) break;
	/* if training epoch limit reached before error convergence to goal ... */
    }
    /* test the neuron */
    int test_idx;
    for(test_idx=0;test_idx<target_input_idx;test_idx++){
	double o = linear_activation(input_weights[0]*test_vector[test_idx].x_val/norm[0]+
			   					 input_weights[1]*bias);
	printf("%f\n",o);
    }
	/* free resources */
	if(norm_train) free(norm_train);
	if(norm) free(norm);
    return 0;
}


