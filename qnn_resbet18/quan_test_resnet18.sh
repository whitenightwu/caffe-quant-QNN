# BASE_PATH=$(pwd)
# TRAIN_PATH=$(dirname $0)

# cd $TRAIN_PATH
# mkdir model

# TOOLS=$BASE_PATH/caffe/build/tools

../.././build/tools/caffe test \
    -model quan_resnet18_train_test.prototxt \
    -weights resnet18_origine.caffemodel \
    -gpu 0  \
    -iterations 1000

