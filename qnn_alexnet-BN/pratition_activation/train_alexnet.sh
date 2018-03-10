../.././build/tools/caffe train \
    -solver quan_solver.prototxt \
    -snapshot ../other_model/alexnet_bit_pratition_iter_10000.solverstate \
    -gpu 1
#    -weights alexnet_origine.caffemodel \
#    -snapshot ../other_model/alexnet_bit_pratition_iter_10000.solverstate
