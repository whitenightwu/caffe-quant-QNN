# BASE_PATH=$(pwd)
# TRAIN_PATH=$(dirname $0)

# cd $TRAIN_PATH
# mkdir model

# TOOLS=$BASE_PATH/caffe/build/tools

.././build/tools/caffe test \
    -model quan_train_val.prototxt \
    -weights alexnet_origine.caffemodel \
    -gpu 1  \
    -iterations 250

