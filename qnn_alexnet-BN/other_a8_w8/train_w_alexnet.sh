../.././build/tools/caffe train \
    -solver quan_aw_solver.prototxt \
    -weights alexnet_origine.caffemodel \
    -gpu 1
#    -weights alexnet_origine.caffemodel \
#    -snapshot ../other_model/alexnet_a8w4_iter_16006.solverstate \
