from imp import reload

import classification as cls

def run_classification():
    reload(cls)
    cls.training_file_name = './data/classification_data.txt'
    cls.groundtruth_file_name = './data/classification_groundtruth.txt'
    cls.run()


if __name__ == '__main__':
    run_classification()

