../.././build/tools/caffe train \
    -solver quan_aw_solver.prototxt \
    -weights alexnet_origine.caffemodel \
    -gpu 0
#    -weights alexnet_origine.caffemodel \
#    -snapshot ../other_model/alexnet_a4w8_iter_2000.solverstate \
