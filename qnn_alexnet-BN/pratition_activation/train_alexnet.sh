
../.././build/tools/caffe train \
    -solver quan_solver.prototxt \
    -snapshot ../other_model/alexnet_activation_iter_1746.solverstate \
    -gpu 0


#    -weights alexnet_origine.caffemodel \
#    -snapshot ../other_model/alexnet_activation_iter_1746.solverstate
