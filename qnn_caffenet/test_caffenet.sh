./build/tools/caffe test \
    -model qnn_caffenet/v1_train_quantized_caffenet.prototxt  \
    -weights qnn_caffenet/bvlc_reference_caffenet.caffemodel   \
    -iterations 50 \
    -gpu 0
# ./build/tools/caffe test \
#     -model qnn_try1/quantized_caffenet.prototxt  \
#     -weights qnn_try1/bvlc_reference_caffenet.caffemodel   \
#     -iterations 1 
#     > ydwu/tmp.log 2>&1



