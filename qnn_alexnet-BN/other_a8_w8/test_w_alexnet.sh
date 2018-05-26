

../.././build/tools/caffe test \
    -model quan_aw_train_val.prototxt \
    -weights ../other_model/alexnet_a8_w8_iter_5000.caffemodel \
    -gpu 1 \
    -iterations 250

#    -weights alexnet_origine.caffemodel \

