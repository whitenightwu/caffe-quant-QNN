./build/tools/caffe test \
    -model qnn_try1/quantized_caffenet.prototxt  \
    -weights qnn_try1/bvlc_reference_caffenet.caffemodel   \
    -iterations 1 \
    > ydwu/tmp.log 2>&1

#    nohup sh ./examples/INQ/alexnet/train_alexnet.sh >run${count}_log.out 2>&1


#./Build/tools/caffe train \
#    --solver=./models/bvlc_reference_caffenet/solver.prototxt  \
#    --weights=./qnn_try1/bvlc_reference_caffenet.caffemodel   \
#    --gpu 1

#./build/tools/caffe train \
#    --solver=./examples/INQ/alexnet/solver.prototxt \
#    --weights=./models/bvlc_alexnet/original.caffemodel \
#    --gpu 1

