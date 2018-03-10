
../.././build/tools/caffe test \
    -model quan_train_val.prototxt \
    -weights ../other_model/alexnet_bit_pratition_iter_10000.caffemodel \
    -gpu 1  \
    -iterations 250

#    -weights alexnet_origine.caffemodel \
