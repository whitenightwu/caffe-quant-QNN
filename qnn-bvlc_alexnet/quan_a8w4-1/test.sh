

../.././build/tools/caffe test \
    -model alexnet_train_val.prototxt \
    -weights ./new-best_model/caffe_alexnet_quan_a8w8_iter_1.caffemodel \
    -gpu 3  \
    -iterations 1000
