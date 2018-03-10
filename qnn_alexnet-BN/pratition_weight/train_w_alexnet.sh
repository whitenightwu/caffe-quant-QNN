../.././build/tools/caffe train \
    -solver quan_w_solver.prototxt \
    -snapshot ../model/alexnet_w_45_iter_20000.solverstate \
    -gpu 1
#    -weights alexnet_origine.caffemodel \
