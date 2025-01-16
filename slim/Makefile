CROSS_COMPILE=
CC=$(CROSS_COMPILE)gcc

CFLAGS = -lm -O3 -Wall -Wextra

all : singlec netc singler netr netrnoise

singlec : p1.c
        $(CC) -o single_neuron_classifier p1.c $(CFLAGS)

netc : p2.c
        $(CC) -o neural_net_classifier p2.c $(CFLAGS)

singler : p3.c
        $(CC) -o single_neuron_regressor p3.c $(CFLAGS)

netr : p4.c
        $(CC) -o neural_net_regressor p4.c $(CFLAGS)

netrnoise : p5.c
        $(CC) -o neural_net_regressor_noise p5.c $(CFLAGS)

clean:
        rm single_neuron_*
        rm neural_net_*
