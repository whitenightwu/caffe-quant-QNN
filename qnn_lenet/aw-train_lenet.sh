

../.././build/tools/caffe train -solver quan_lenet_solver-aw.prototxt -weights lenet_origine.caffemodel  -gpu 1 \
    > tmp_aw-4bit-train-lenet.log  2>&1

# ../.././build/tools/caffe train \
#     -solver=./quan_lenet_solver.prototxt  \
#     -weights=./lenet_origine.caffemodel  \
#     --gpu 1
