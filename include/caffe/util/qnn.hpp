#ifndef CAFFE_UTIL_QNN_H_
#define CAFFE_UTIL_QNN_H_


#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

  template <typename Dtype> Dtype Weight_Quantization(Dtype weights);

  void Linear_quant(double& scaling_factor, double& min_value, double& max_value);

}
#endif
