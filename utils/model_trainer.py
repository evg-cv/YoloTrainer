from darkflow.net.build import TFNet


if __name__ == '__main__':

    options = {"model": "/media/mensa/Data/Task/EgyALPR/darkflow/cfg/yolo.cfg",
               "load": "/media/mensa/Data/Task/EgyALPR/darkflow/bin/yolo.weights",
               "batch": 8,
               "epoch": 100,
               "gpu": 0.9,
               "train": True,
               "annotation": "/media/mensa/Data/Task/EgyALPR/training_dataset/xml",
               "dataset": "/media/mensa/Data/Task/EgyALPR/training_dataset/images"}
    tf_net = TFNet(options)
    tf_net.train()
    tf_net.savepb()
