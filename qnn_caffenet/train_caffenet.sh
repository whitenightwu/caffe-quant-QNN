

# import os

# #################################
# os.system("sed -i \"s/(bit_width: 4)/(bit_width: 5)/g\" ./qnn_try1/4bit-v1_train_quantized_caffenet.prototxt")
# #    bit_width: 4


./build/tools/caffe train \
    -solver=./qnn_caffenet/solver.prototxt  \
    -weights=./qnn_caffenet/bvlc_reference_caffenet.caffemodel  \
    --gpu 1
#    > ydwu/train.log 2>&1
    

#./build/tools/caffe train \
#    --solver=./examples/INQ/alexnet/solver.prototxt \
#    --weights=./models/bvlc_alexnet/original.caffemodel \
#    --gpu 1

