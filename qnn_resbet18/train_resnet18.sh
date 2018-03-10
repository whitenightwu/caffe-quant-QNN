
#error
BASE_PATH=$(pwd)
TRAIN_PATH=$(dirname $0)

cd $TRAIN_PATH
mkdir model

TOOLS=$BASE_PATH/caffe/build/tools

$TOOLS/caffe train \
    -solver solver.prototxt \
    -weights resnet18_origine.caffemodel \
    -gpu 0

