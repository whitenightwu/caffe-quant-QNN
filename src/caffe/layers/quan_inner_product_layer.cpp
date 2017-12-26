#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/quan_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>
#include <algorithm>
#include <limits>
#include <typeinfo>

namespace caffe {

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.quan_inner_product_param().num_output();
  bias_term_ = this->layer_param_.quan_inner_product_param().bias_term();
  transpose_ = this->layer_param_.quan_inner_product_param().transpose();
  N_ = num_output;
  LOG(INFO) << num_output << "   " << N_;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.quan_inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N quan_inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.quan_inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.quan_inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  /***********************get paramerats*************************/
  // parses the parameters from *.prototxt and quick sanity check
  bit_width_ = this->layer_param_.quan_inner_product_param().bit_width();
  CHECK_GT(bit_width_, 0) << type() << " Layer has unexpected negative bit width";

  round_method_ = this->layer_param_.quan_inner_product_param().round_method();
  round_strategy_ = this->layer_param_.quan_inner_product_param().round_strategy();

  // read range
  range_low_ = this->layer_param_.quan_inner_product_param().range_low();
  range_high_ = this->layer_param_.quan_inner_product_param().range_high();
}

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.quan_inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with quan_inner product parameters.";
  // The first "axis" dimensions are independent quan_inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

  template <typename Dtype>
void QuanInnerProductLayer<Dtype>::Weight_Quantization(Dtype& weights)
  {
    Dtype scaling_factor = 0;
    Dtype min_value = 0;
    Dtype max_value = 0;

    /******************************************/

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
    Dtype neg_scaling_factor = (range_low_ < 0) ? log2(min_value/range_low_) :
      std::numeric_limits<Dtype>::infinity();
    Dtype pos_scaling_factor = (range_high_ > 0) ? log2(max_value/range_high_) :
      std::numeric_limits<Dtype>::infinity();
    
    switch (round_strategy_)
      {
      case QuanInnerProductParameter_RoundStrategy_CONSERVATIVE:
	scaling_factor = pow(2.0, floor(std::min(neg_scaling_factor, pos_scaling_factor)));
	break;
      case QuanInnerProductParameter_RoundStrategy_NEUTRAL:
	scaling_factor = pow(2.0, round(std::min(neg_scaling_factor, pos_scaling_factor)));
	break;
      case QuanInnerProductParameter_RoundStrategy_AGGRESSIVE:
	scaling_factor = pow(2.0, ceil(std::min(neg_scaling_factor, pos_scaling_factor)));
	break;
      default:
	LOG(FATAL) << "Unknown round strategy.";
      }
    /******************************************/

    Dtype weight_rounded;
     
    switch (round_method_) 
      {
      case QuanInnerProductParameter_RoundMethod_ROUND:
	weight_rounded = round(weights * (Dtype)scaling_factor);
	break;
      case QuanInnerProductParameter_RoundMethod_FLOOR:
	weight_rounded = floor(weights * (Dtype)scaling_factor);
	break;
      case QuanInnerProductParameter_RoundMethod_CEIL:
	weight_rounded = ceil(weights * (Dtype)scaling_factor);
	break;
      case QuanInnerProductParameter_RoundMethod_TRUNC:
	weight_rounded = trunc(weights * (Dtype)scaling_factor);
	break;
      default:
	LOG(FATAL) << "Unknown round method.";
      }
    
    weight_rounded = floor(weights * (Dtype)scaling_factor);
    // y = clip(x, min, max) / scaling_factor; so y in [min/scaling_factor, max/scaling_factor]
    weights = std::min(std::max((Dtype)weight_rounded, (Dtype)(min_value)), (Dtype)(max_value)) /
      (Dtype)(scaling_factor);
  }


  /*
template <typename Dtype>
void analyze_scaling_factor(double& scaling_factor,
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
	double neg_scaling_factor = (range_low_ < 0) ? log2(min_value/range_low_) :
			std::numeric_limits<double>::infinity();
	double pos_scaling_factor = (range_high_ > 0) ? log2(max_value/range_high_) :
			std::numeric_limits<double>::infinity();

	switch (round_strategy_) {
	case QuanInnerProductParameter_RoundStrategy_CONSERVATIVE:
		scaling_factor = pow(2.0, floor(std::min(neg_scaling_factor, pos_scaling_factor)));
		break;
	case QuanInnerProductParameter_RoundStrategy_NEUTRAL:
		scaling_factor = pow(2.0, round(std::min(neg_scaling_factor, pos_scaling_factor)));
		break;
	case QuanInnerProductParameter_RoundStrategy_AGGRESSIVE:
		scaling_factor = pow(2.0, ceil(std::min(neg_scaling_factor, pos_scaling_factor)));
		break;
	default:
		LOG(FATAL) << "Unknown round strategy.";
	}

}


template <typename Dtype>
Dtype fixed_point(const Dtype& input_data,
		const double& scaling_factor, const double& min_value,
		const double& max_value) const {
	// rounded results of input data
	double input_data_rounded;

	switch (round_method_) {
	case QuanInnerProductParameter_RoundMethod_ROUND:
		input_data_rounded = round(input_data * (Dtype)scaling_factor);
		break;
	case QuanInnerProductParameter_RoundMethod_FLOOR:
		input_data_rounded = floor(input_data * (Dtype)scaling_factor);
		break;
	case QuanInnerProductParameter_RoundMethod_CEIL:
		input_data_rounded = ceil(input_data * (Dtype)scaling_factor);
		break;
	case QuanInnerProductParameter_RoundMethod_TRUNC:
		input_data_rounded = trunc(input_data * (Dtype)scaling_factor);
		break;
	default:
		LOG(FATAL) << "Unknown round method.";
	}

	return std::min(std::max(input_data_rounded, min_value), max_value) /
			(Dtype)scaling_factor;
}

*/





template <typename Dtype>
void QuanInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  /***********************get range*************************/
  //  LOG(INFO) << "range_high_ =" << range_high_ << ";range_low_ =" << range_low_;
  Dtype* tmp_weight = (Dtype*) malloc((this->blobs_[0]->count())*sizeof(Dtype));
  caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(), tmp_weight);
  Dtype* Q_weight = const_cast<Dtype*>(tmp_weight);
  // get range_high_ and range_low_.
  if(range_high_ == range_low_ )
    {
      Dtype* sort_weight = tmp_weight;
      int qcount_ = this->blobs_[0]->count();
      std::sort(sort_weight, sort_weight+(this->blobs_[0]->count()));
      range_high_ = sort_weight[qcount_-1];
      range_low_ = sort_weight[0];
    }
   LOG(INFO) << "range_high_ =" << range_high_ << ";range_low_ =" << range_low_;


  /***********************quantized*************************/
  ////////////////////////origine///////////////////////
  /*
  double scaling_factor, min_value, max_value;
  analyze_scaling_factor(scaling_factor, min_value, max_value);

  // apply quantization element-wise
  for (int i = 0; i < (this->blobs_[0]->count()); ++i) {
    Q_weight[i] = fixed_point(Q_weight[i], scaling_factor,
			      min_value, max_value);
  }
  */

  for (int i = 0; i < (this->blobs_[0]->count()); ++i) 
    {
      Weight_Quantization(*(Q_weight+i));
    }
  const Dtype *weight = Q_weight;

  /***********************print*************************/
  //  std::cout << "fc_max:" << range_high_ << "  " << "fc_min:" << range_low_ << std::endl;
  // for (int i = 0; i < 5; ++i) 
  //   {
  //     std::cout << "weight:" << this->blobs_[0]->cpu_data()[i] << std::endl;
  //     std::cout << "qnn_weight:" << weight[i] << std::endl;
  //   }
 
  /*****************************************************/
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(QuanInnerProductLayer);
#endif

  INSTANTIATE_CLASS(QuanInnerProductLayer);
  REGISTER_LAYER_CLASS(QuanInnerProduct);

}  // namespace caffe
