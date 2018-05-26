

# ../.././build/tools/caffe test \
#     -model train_val.prototxt \
#     -weights ../quan_a8w4-1/new-best_model/caffe_alexnet_quan_a8w8_iter_1.caffemodel \
#     -gpu 3  \
#     -iterations 1000


../.././build/tools/caffe test \
    -model train_val.prototxt \
    -weights ../bvlc_alexnet.caffemodel \
    -gpu 3  \
    -iterations 1000
