../.././build/tools/caffe train \
    -solver quan_aw_solver.prototxt \
    -weights ../alexnet_origine.caffemodel \
    -gpu 0

#    -snapshot ./model/alexnet_bit_pratition_iter_60000.solverstate \

