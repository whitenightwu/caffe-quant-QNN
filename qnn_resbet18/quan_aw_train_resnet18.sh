
../.././build/tools/caffe train \
    -solver quan_resnet18_aw_solver.prototxt \
    -weights resnet18_origine.caffemodel \
    -gpu 0 
#    > 8bit-act.log 2>&1



# BASE_PATH=$(pwd)
# TRAIN_PATH=$(dirname $0)

# cd $TRAIN_PATH
# mkdir model

# TOOLS=$BASE_PATH/caffe/build/tools

