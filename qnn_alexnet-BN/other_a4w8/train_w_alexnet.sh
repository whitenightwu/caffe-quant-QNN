../.././build/tools/caffe train \
    -solver quan_aw_solver.prototxt \
    -weights alexnet_origine.caffemodel \
    -gpu 3
#    -weights alexnet_origine.caffemodel \
#    -snapshot ../model/alexnet_w_45_iter_20000.solverstate \
