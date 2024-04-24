/* 
Single Neuron for Classification
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>

/* max set length */
#define SET_LEN 1000

/* performance/goal value */
#define GOAL 0.0

/* max number of epochs for training */
#define MAX_EPOCHS 1000

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
    return (float)rand() / (float)RAND_MAX;
}

/* the linear activation function */
int linear_activation(double in){
    return (in >= 0) ? 1 : -1;
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

/* get the decision boundry - parametrization for "best weights" */
void get_decision_boundry(double input_weights[3]){
	printf("\nDecision boundary equation: %.2f*x + %.2f*y + %.2f = 0\n", 
		   input_weights[0], input_weights[1], input_weights[2]);
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
    /* valid input values counter limited to SETLEN */
    int input_idx=0;
    /* valid input values counter limited to SETLEN for target values */
    int target_input_idx=0;
    /* training vector */
    struct training_set train_vector[SET_LEN];
    /* testing vector */
    struct testing_set test_vector[SET_LEN];
    /* input weights*/
    double input_weights[3];
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
    int bias=-1;
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

#if 0
    show_training_vector(train_vector,input_idx);
#endif

    /* reinit index */
    target_input_idx=0;
    /* the next input values are test/target values to validate against */
    while(scanf("%lf,%lf\n", &test_x_val, &test_y_val)>0){
            /* populate the testing set */
            test_vector[target_input_idx].x_val = test_x_val;
            test_vector[target_input_idx].y_val = test_y_val;
			/* additional storage for computing max - used in normalization sequence */
			x_buffer_test[target_input_idx] = test_x_val;
			y_buffer_test[target_input_idx] = test_y_val;
            /* check if max input reached */
      if(target_input_idx==1000) break;
      target_input_idx++;
    }

	/* get the normalization value for test set */
	norm_test = normalize(x_buffer_test, y_buffer_test, target_input_idx);

	/* get the unified normalization values */
	if(norm_test[0]>norm_train[0]) norm[0] = norm_test[0];
	else norm[0] = norm_train[0];
	if(norm_test[1]>norm_train[1]) norm[1] = norm_test[1];
	else norm[1] = norm_train[1];

#if 0
    show_testing_vector(test_vector, target_input_idx);
#endif

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
    for(epochs=0;epochs<MAX_EPOCHS;epochs++){
        /* iterate through training values */
        for(train_idx=0;train_idx<get_training_vector_length(train_vector);train_idx++){
            /* compute sum */
            sum=input_weights[0]*train_vector[train_idx].x_val/norm[0]+
                input_weights[1]*train_vector[train_idx].y_val/norm[1]+
				input_weights[2]*bias;
            /* compute the output */
            net_out = linear_activation(sum);
            /* compute delta between current output and the target value */
            delta=train_vector[train_idx].out - net_out;

			/* update the weights */
            input_weights[0]+=eta*delta*train_vector[train_idx].x_val/norm[0];
            input_weights[1]+=eta*delta*train_vector[train_idx].y_val/norm[1];
            input_weights[2]+=eta*delta*bias;
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
	int o = linear_activation(input_weights[0]*test_vector[test_idx].x_val/norm[0]+
							  input_weights[1]*test_vector[test_idx].y_val/norm[1]+
							  input_weights[2]*bias);
	if(o==1) printf("+%d\n",o);
	else printf("%d\n",o);
    }
	/* free resources */
	if(norm_train) free(norm_train);
	if(norm_test) free(norm_test);
	if(norm) free(norm);
    return 0;
}


