

../.././build/tools/caffe test \
    -model quan_aw_train_val.prototxt \
    -weights alexnet_a4w8_iter_2000.caffemodel \
    -gpu 3 \
    -iterations 250

#    -weights alexnet_origine.caffemodel \
#    -snapshot ../model/alexnet_w_45_iter_20000.solverstate \
