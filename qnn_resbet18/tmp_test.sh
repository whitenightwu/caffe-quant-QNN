# BASE_PATH=$(pwd)
# TRAIN_PATH=$(dirname $0)

# cd $TRAIN_PATH
# mkdir model

# TOOLS=$BASE_PATH/caffe/build/tools

../.././build/tools/caffe test \
    -model tmp_train_test.prototxt \
    -weights tmp_origine.caffemodel \
    -gpu 0  \
    -iterations 1000

