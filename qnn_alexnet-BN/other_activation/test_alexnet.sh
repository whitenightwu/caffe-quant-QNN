# BASE_PATH=$(pwd)
# TRAIN_PATH=$(dirname $0)

# cd $TRAIN_PATH
# mkdir model

# TOOLS=$BASE_PATH/caffe/build/tools

../.././build/tools/caffe test \
    -model quan_train_val.prototxt \
    -weights ../other_model/alexnet_bit_pratition_iter_8000.caffemodel \
    -gpu 1  \
    -iterations 250

#    -weights ../alexnet_origine.caffemodel \
