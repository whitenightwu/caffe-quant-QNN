

../.././build/tools/caffe test \
    -model test.prototxt \
    -weights alexnet_cvgj_iter_320000.caffemodel \
    -gpu 3 \
    -iterations 1000

#    -weights alexnet_origine.caffemodel \
#    -snapshot ../model/alexnet_w_45_iter_20000.solverstate \
