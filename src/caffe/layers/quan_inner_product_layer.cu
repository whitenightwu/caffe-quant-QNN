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
__host__ void analyze_scaling_factor_gpu(Dtype& scaling_factor, Dtype& min_value, Dtype& max_value, int bit_width_,  float range_low_,float range_high_, QuanInnerProductParameter_RoundStrategy round_strategy_)  {
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
      0;
	double pos_scaling_factor = (range_high_ > 0) ? log2(max_value/range_high_) :
      0;

	switch (round_strategy_) {
	case QuantizationParameter_RoundStrategy_CONSERVATIVE:
		scaling_factor = pow(Dtype(2.0), Dtype(floor(min(neg_scaling_factor, pos_scaling_factor))));
		break;
	case QuantizationParameter_RoundStrategy_NEUTRAL:
		scaling_factor = pow(Dtype(2.0), Dtype(round(min(neg_scaling_factor, pos_scaling_factor))));
		break;
	case QuantizationParameter_RoundStrategy_AGGRESSIVE:
		scaling_factor = pow(Dtype(2.0), Dtype(ceil(min(neg_scaling_factor, pos_scaling_factor))));
		break;
	default:
;
	//	LOG(FATAL) << "Unknown round strategy.";
	}
//LOG(INFO) << " scaling_factor" << scaling_factor << " pos_scaling_factor" << pos_scaling_factor << "  neg_scaling_factor" << neg_scaling_factor;

}

template <typename Dtype>
__host__ void fixed_point_gpu(Dtype& weight_in, QuanInnerProductParameter_RoundMethod round_method_, const double &scaling_factor, const double &min_value, const double &max_value) {

	double weight_rounded;

	switch (round_method_) {
	case QuantizationParameter_RoundMethod_ROUND:
		weight_rounded = round(weight_in * (Dtype)scaling_factor);
		break;
	case QuantizationParameter_RoundMethod_FLOOR:
		weight_rounded = floor(weight_in * (Dtype)scaling_factor);
		break;
	case QuantizationParameter_RoundMethod_CEIL:
		weight_rounded = ceil(weight_in * (Dtype)scaling_factor);
		break;
	case QuantizationParameter_RoundMethod_TRUNC:
		weight_rounded = trunc(weight_in * (Dtype)scaling_factor);
		break;
	default:
;
//		LOG(FATAL) << "Unknown round method.";
	}

    weight_in = min(max((Dtype)weight_rounded, (Dtype)(min_value)), (Dtype)(max_value)) / (Dtype)(scaling_factor);
}


template <typename Dtype>
__host__ void inn_DoReFa_gpu(Dtype& weight_in, QuanInnerProductParameter_RoundMethod round_method_, const double &scaling_factor, const double &tanh_weight_max) {

	weight_in = tanh(weight_in) / (Dtype)tanh_weight_max + (Dtype)0.5;

	switch (round_method_) {
	case QuantizationParameter_RoundMethod_ROUND:
		weight_in = round(weight_in * (Dtype)scaling_factor) / scaling_factor;
		break;
	case QuantizationParameter_RoundMethod_FLOOR:
		weight_in = floor(weight_in * (Dtype)scaling_factor) / scaling_factor;
		break;
	case QuantizationParameter_RoundMethod_CEIL:
		weight_in = ceil(weight_in * (Dtype)scaling_factor) / scaling_factor;
		break;
	case QuantizationParameter_RoundMethod_TRUNC:
		weight_in = trunc(weight_in * (Dtype)scaling_factor) / scaling_factor;
		break;
	default:
		LOG(FATAL) << "Unknown round method.";
	}
}


  /******************************************************************************/
  /*******************************Forward_gpu************************************/
  /******************************************************************************/

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  /***********************get range*************************/
    Dtype* tmp_weight = (Dtype*) malloc((this->blobs_[0]->count())*sizeof(Dtype));
    caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), tmp_weight);
//LOG(INFO) << tmp_weight[0] << "++++tmp" << tmp_weight[1];

//  Dtype* tmp_weight = (Dtype*) malloc((this->blobs_[0]->count())*sizeof(Dtype));
//  caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), tmp_weight);

  Dtype* Q_weight = const_cast<Dtype*>(tmp_weight);
//LOG(INFO) << Q_weight[0] << "++++Q_weight" << Q_weight[1];
  // get range_high_ and range_low_.
  // if(is_runtime_)
  //   {
  //     Dtype* sort_weight = tmp_weight;
  //     int qcount_ = this->blobs_[0]->count();
  //     std::sort(sort_weight, sort_weight+(this->blobs_[0]->count()));
  //     range_high_ = sort_weight[qcount_-1];
  //     range_low_ = sort_weight[0];
  //   }

  /***********************QNN-quantized*************************/
    Dtype scaling_factor =0;
    Dtype min_value=0 ;
    Dtype max_value=0;

//LOG(INFO) << "bit_width=" << bit_width_ << ";  round_method=" << round_method_ << ";  round_strategy=" << round_strategy_ << ";  is_runtime=" << is_runtime_ << ";  range_low=" << range_low_ << ";  range_high=" << range_high_ ;



    analyze_scaling_factor_gpu(scaling_factor, min_value, max_value, bit_width_, range_low_,range_high_, round_strategy_);
//LOG(INFO) << " scaling_factor" << scaling_factor << " min_value" << min_value << "   max_value" << max_value;

    for (int i = 0; i < (this->blobs_[0]->count()); ++i) 
      {
	fixed_point_gpu( *(Q_weight+i),  round_method_,  scaling_factor, min_value, max_value);
      }


//	Weight_Quantization_gpu(*(Q_weight+i));
//    const Dtype *weight = Q_weight;
//copy_weight_gpu<<<1,1>>>(Q_weight, weight);

const Dtype *in_weight = Q_weight;
Dtype *A_weight;
cudaMalloc((void**)&A_weight, this->blobs_[0]->count() * sizeof(Dtype));
caffe_gpu_memcpy(this->blobs_[0]->count() * sizeof(Dtype), in_weight, A_weight);
const Dtype *weight = A_weight;


  /***********************DoReFa-quantized*************************/



//  Dtype weight_max = max(-range_low_, range_high_);
//  Dtype tanh_weight_max = 2 * tanh(weight_max);
//   // Dtype weight_max = range_high_ - range_low_;
//   // Dtype tanh_weight_max = tanh(weight_max);
//   Dtype scaling_factor = pow((Dtype)2.0, (Dtype)bit_width_) - 1;

// LOG(INFO) << "tanh_weight_max=" << tanh_weight_max << "  weight_max=" << weight_max  << "  scaling_factor=" << scaling_factor;
// LOG(INFO) << Q_weight[0];

//   for (int i = 0; i < (this->blobs_[0]->count()); ++i) 
//   {
//     inn_DoReFa_gpu( *(Q_weight+i),  round_method_,  scaling_factor, tanh_weight_max);
//   }
// LOG(INFO) << Q_weight[0];
// const Dtype *in_weight = Q_weight;
// Dtype *A_weight;
// cudaMalloc((void**)&A_weight, this->blobs_[0]->count() * sizeof(Dtype));
// caffe_gpu_memcpy(this->blobs_[0]->count() * sizeof(Dtype), in_weight, A_weight);
// const Dtype *weight = A_weight;





  /***********************print*************************/
  //  std::cout << "fc_max:" << range_high_ << "  " << "fc_min:" << range_low_ << std::endl;
  // for (int i = 0; i < 5; ++i) 
  //   {
  //     std::cout << "weight:" << this->blobs_[0]->gpu_data()[i] << std::endl;
  //     std::cout << "qnn_weight:" << weight[i] << std::endl;
  //   }
 
  /***********************compute*************************/
  caffe_gpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(),
        this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }

cudaFree(A_weight);
free(tmp_weight);
//cudaFree(weight);
}


template <typename Dtype>
void QuanInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(QuanInnerProductLayer);

}  // namespace caffe
