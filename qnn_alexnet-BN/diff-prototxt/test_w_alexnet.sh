# BASE_PATH=$(pwd)
# TRAIN_PATH=$(dirname $0)

# cd $TRAIN_PATH
# mkdir model

# TOOLS=$BASE_PATH/caffe/build/tools
#    -gpu 1 \


../.././build/tools/caffe test \
    -model quan_w_train_val.prototxt \
    -weights ../alexnet_origine.caffemodel \
    -gpu 1 \
    -iterations 250

