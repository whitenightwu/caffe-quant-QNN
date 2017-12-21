/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : quan_linear.cpp
 * Authors    : ydwu@aries
 * Create Time: 2017-12-19:15:56:55
 * Description:
 * 
 */
#include "caffe/util/qnn.hpp"
#include <iostream>

#include <algorithm>
#include <vector>
#include <limits>


namespace caffe
{
  template <typename Dtype>
  void Linear_quant(double& scaling_factor, double& min_value, double& max_value)
  {
    // defind range_low and range_high
    double range_low_ = -0.8;
    double range_high_ = 0.8;
    double bit_width_ = 4;

    // smart choosing between 2s complement encoding or unsigned encoding
    if (range_low_ >= 0.0) {
      // non-negative input range with unsigned range [0, 2^N-1]
      min_value = 0.0;
      max_value = pow(2.0, bit_width_) - 1.0;
    } else if (range_high_ <= 0.0) {
      // non-positive input range with unsigned range [-2^N+1, 0]
      min_value = -pow(2.0, bit_width_) + 1.0;
      max_value = 0.0;
    } else {
      // N-bit 2s complement can represent the integer between -2^(N-1)
      // to 2^(N-1)-1
      min_value = -pow(2.0, bit_width_-1);
      max_value = pow(2.0, bit_width_-1) - 1.0;
    }

    // analyze the scaling factor based on min(max)value and range
    // scaling factor should be power of 2
    // example:  scaling_factor = 2^(round(X)); X = log2(min_value / range_low), in [0,1]
    double neg_scaling_factor = (range_low_ < 0) ? log2(min_value/range_low_) :
      std::numeric_limits<double>::infinity();
    double pos_scaling_factor = (range_high_ > 0) ? log2(max_value/range_high_) :
      std::numeric_limits<double>::infinity();
    scaling_factor = pow(2.0, round(std::min(neg_scaling_factor, pos_scaling_factor)));
    /*
      switch (round_strategy_) {
      case QuantizationParameter_RoundStrategy_CONSERVATIVE:
      scaling_factor = pow(2.0, floor(std::min(neg_scaling_factor, pos_scaling_factor)));
      break;
      case QuantizationParameter_RoundStrategy_NEUTRAL:
      scaling_factor = pow(2.0, round(std::min(neg_scaling_factor, pos_scaling_factor)));
      break;
      case QuantizationParameter_RoundStrategy_AGGRESSIVE:
      scaling_factor = pow(2.0, ceil(std::min(neg_scaling_factor, pos_scaling_factor)));
      break;
      default:
      LOG(FATAL) << "Unknown round strategy.";
      }
    */
  }

  template <typename Dtype>
  double Weight_Quantization(Dtype weights)
  {
    // rounded results of input data
    double* scaling_factor;
    double* min_value;
    double* max_value;
    double input_data_rounded;

    Linear_quant(*scaling_factor, *min_value, *max_value);

    /*
      switch (round_method_) {
      case QuantizationParameter_RoundMethod_ROUND:
      input_data_rounded = round(input_data * (Dtype)scaling_factor);
      break;
      case QuantizationParameter_RoundMethod_FLOOR:
      input_data_rounded = floor(input_data * (Dtype)scaling_factor);
      break;
      case QuantizationParameter_RoundMethod_CEIL:
      input_data_rounded = ceil(input_data * (Dtype)scaling_factor);
      break;
      case QuantizationParameter_RoundMethod_TRUNC:
      input_data_rounded = trunc(input_data * (Dtype)scaling_factor);
      break;
      default:
      LOG(FATAL) << "Unknown round method.";
      }
    */
    Dtype weight_rounded;
    weight_rounded = round((double)weights * (*scaling_factor));
 
    // y = clip(x, min, max) / scaling_factor; so y in [min/scaling_factor, max/scaling_factor]
    return std::min(std::max(weight_rounded, min_value), max_value) /
      (Dtype)(*scaling_factor);
  }
  
  template <typename Dtype> Dtype Weight_Quantization(Dtype weights);
  void Linear_quant(double& scaling_factor, double& min_value, double& max_value);
  
}
