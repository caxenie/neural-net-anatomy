from imp import reload

import regression as reg

def run_regression():
    reload(reg)
    reg.training_file_name = './data/regression_data.txt'
    reg.groundtruth_file_name = './data/regression_groundtruth.txt'
    reg.run()


if __name__ == '__main__':
    run_regression()
