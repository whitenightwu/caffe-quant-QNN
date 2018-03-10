#include <algorithm>
#include <vector>
#include <limits>


#include "caffe/filler.hpp"
#include "caffe/layers/quantization_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void QuantizationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	// analyze the scaling factor, accompanied with min/max range of data
	double scaling_factor, min_value, max_value;
	analyze_scaling_factor_gpu(scaling_factor, min_value, max_value);

	Dtype aaa;
	// apply quantization element-wise
	for (int i = 0; i < count; ++i) 
	{
	//  top_data[i] = fixed_point(bottom_data[i], scaling_factor, min_value, max_value);
//LOG(INFO) << "method" << round_method_; result: round_method_ = 1
	switch (round_method_) {
	case QuantizationParameter_RoundMethod_ROUND:
		aaa = round(bottom_data[i] * (Dtype)scaling_factor);
//LOG(INFO) << "ok1";
		break;
	case QuantizationParameter_RoundMethod_FLOOR:
		aaa = (Dtype)floor(bottom_data[i] * (Dtype)scaling_factor);
LOG(INFO) << "ok2::" << aaa; 
		break;
	case QuantizationParameter_RoundMethod_CEIL:
		aaa = ceil(bottom_data[i] * (Dtype)scaling_factor);
LOG(INFO) << "ok3";
		break;
	case QuantizationParameter_RoundMethod_TRUNC:
		aaa = trunc(bottom_data[i] * (Dtype)scaling_factor);
LOG(INFO) << "ok4";
		break;
	default:
		LOG(FATAL) << "Unknown round method.";
	}
//LOG(INFO) << "min_value" << min_value; so min_value =-128
//LOG(INFO) << "input_data_rounded" << input_data_rounded;
	//top_data[i] = (Dtype)std::min(std::max(input_data_rounded, min_value), max_value) / (Dtype)scaling_factor;
	}

LOG(INFO) << "ydwu============";
}
				
template <typename Dtype>
void QuantizationLayer<Dtype>::analyze_scaling_factor_gpu(double& scaling_factor,
		double& min_value, double& max_value) const {
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
}

template <typename Dtype>
Dtype QuantizationLayer<Dtype>::fixed_point_gpu(const Dtype& input_data,
		const double& scaling_factor, const double& min_value,
		const double& max_value) const {
	// rounded results of input data
	double input_data_rounded;
LOG(INFO) << "1============";
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
LOG(INFO) << "2============";
	// y = clip(x, min, max) / scaling_factor; so y in [min/scaling_factor, max/scaling_factor]
	return std::min(std::max(input_data_rounded, min_value), max_value) /
			(Dtype)scaling_factor;
}


template <typename Dtype>
void QuantizationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

  if (!propagate_down[0]) { return; }
  
  const Dtype* top_diff = top[0]->gpu_diff();  
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();  
  const int count = bottom[0]->count();  
  for (int i = 0; i < count; ++i) 
    {  
      bottom_diff[i] = top_diff[i];  
    }

}


INSTANTIATE_LAYER_GPU_FUNCS(QuantizationLayer);

}  // namespace caffe