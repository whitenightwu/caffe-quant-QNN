

../.././build/tools/caffe train \
    -model train.prototxt \
    -weights alexnet_cvgj_iter_320000.caffemodel \
    -gpu 3 \
    -solver solver.prototxt
#    -weights alexnet_origine.caffemodel \
#    -snapshot ../model/alexnet_w_45_iter_20000.solverstate \
