Compile the individual C files or use make with the provided Makefile

To execute simply provide the training file as stream input and log the output in an output stream (i.e., a file).

In a terminal after compilation simply run:

./neural_net_classifier.exe < classification_data.txt > output-neuron-classifier.txt
./single_neuron_classifier.exe < classification_data.txt > output-net-classifier.txt
./neural_net_regressor.exe < regression_data.txt > output-neuron-regressor.txt
./single_neuron_regressor.exe < regression_data.txt > output-net-regressor.txt
./neural_net_regressor_noise.exe < regression_data.txt > output-neuron-regressor-noise.txt

To check how good the network did learn from the data you can compare the output file with the ground truth (i.e., expected output). Simply install and use Meld as a tool to do diff (i.e., find and mark differences among text files). 

./neural_net_classifier.exe < classification_data.txt > output.txt && meld.exe output.txt classification_groundtruth.txt
./single_neuron_classifier.exe < classification_data.txt > output.txt && meld.exe output.txt classification_groundtruth.txt