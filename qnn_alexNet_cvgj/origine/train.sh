../.././build/tools/caffe train \
    -solver solver.prototxt \
    -weights alexnet_cvgj_iter_320000.caffemodel \
    -gpu 3

# ../.././build/tools/caffe train \
# 			  -solver train.solver \
# 			  -weights alexnet_cvgj_iter_320000.caffemodel \
# 			  -gpu 3 
#    -iterations 1000

#    -weights alexnet_origine.caffemodel \
#    -snapshot ../model/alexnet_w_45_iter_20000.solverstate \


# ../.././build/tools/caffe train \
#     -solver quan_aw_solver.prototxt \
#     -weights alexnet_origine.caffemodel \
#     -gpu 0
# #    -weights alexnet_origine.caffemodel \
# #    -snapshot ../other_model/alexnet_a4w8_iter_2000.solverstate \
