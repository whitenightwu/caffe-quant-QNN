#include <vector>

#include "caffe/layers/quan_conv_layer.hpp"

#include <iostream>
#include <algorithm>
#include <limits>

namespace caffe {


template <typename Dtype>
__host__ void analyze_scaling_factor_gpu(Dtype& scaling_factor, Dtype& min_value, Dtype& max_value, int bit_width_,  float range_low_,float range_high_, QuanConvolutionParameter_RoundStrategy round_strategy_)  {
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
      std::numeric_limits<Dtype>::infinity();
	double pos_scaling_factor = (range_high_ > 0) ? log2(max_value/range_high_) :
      std::numeric_limits<Dtype>::infinity();

	switch (round_strategy_) {
	case QuantizationParameter_RoundStrategy_CONSERVATIVE:
		scaling_factor = pow(Dtype(2.0), Dtype(floor(min(neg_scaling_factor, pos_scaling_factor)))) - 1;
		break;
	case QuantizationParameter_RoundStrategy_NEUTRAL:
		scaling_factor = pow(Dtype(2.0), Dtype(round(min(neg_scaling_factor, pos_scaling_factor)))) - 1;
		break;
	case QuantizationParameter_RoundStrategy_AGGRESSIVE:
		scaling_factor = pow(Dtype(2.0), Dtype(ceil(min(neg_scaling_factor, pos_scaling_factor)))) -1;
		break;
	default:
		LOG(FATAL) << "Unknown round strategy.";
	}
//LOG(INFO) << " scaling_factor" << scaling_factor << " pos_scaling_factor" << pos_scaling_factor << "  neg_scaling_factor" << neg_scaling_factor;

}

template <typename Dtype>
__host__ void fixed_point_gpu(Dtype& weight_in, QuanConvolutionParameter_RoundMethod round_method_, const double &scaling_factor, const double &min_value, const double &max_value) {

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
		LOG(FATAL) << "Unknown round method.";
	}

    weight_in = min(max((Dtype)weight_rounded, (Dtype)(min_value)), (Dtype)(max_value)) / (Dtype)(scaling_factor);

}


// template <typename Dtype>
// __global__ void quan_tanh_norm(const int n, const Dtype* weight) {
//   CUDA_KERNEL_LOOP(index, n) {
//     weight[index] = tanh(fabs(weight[index]));
//   }
// }



  template <typename Dtype>
  void QuanConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
						const vector<Blob<Dtype>*>& top) {
  /***********************get range*************************/
  //  LOG(INFO) << "range_high_ =" << range_high_ << ";range_low_ =" << range_low_;
   // Dtype* tmp_weight;
   // cudaMalloc((void**)&tmp_weight, this->blobs_[0]->count() * sizeof(Dtype));
   // caffe_gpu_memcpy(this->blobs_[0]->count() * sizeof(Dtype), this->blobs_[0]->gpu_data(), tmp_weight);
   Dtype* tmp_weight = (Dtype*) malloc((this->blobs_[0]->count())*sizeof(Dtype));
  caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), tmp_weight);
//LOG(INFO) << tmp_weight[0] << "++++tmp" << tmp_weight[1];

  Dtype* Q_weight = const_cast<Dtype*>(tmp_weight);



  /***********************tanh*************************/
// LOG(INFO) <<  Q_weight[0];
//   const int qcount = this->blobs_[0]->count();
//   quan_tanh_norm<Dtype><<<CAFFE_GET_BLOCKS(qcount), CAFFE_CUDA_NUM_THREADS>>>(
//       qcount, Q_weight);
//   CUDA_POST_KERNEL_CHECK;
// LOG(INFO) <<  Q_weight[0];


  /***********************quantized*************************/
    // double scaling_factor;
    // double min_value;
    // double max_value;
    Dtype scaling_factor =0;
    Dtype min_value=0 ;
    Dtype max_value=0;

//LOG(INFO) << "bit_width=" << bit_width_ << ";  round_method=" << round_method_ << ";  round_strategy=" << round_strategy_ << ";  is_runtime=" << is_runtime_ << ";  range_low=" << range_low_ << ";  range_high=" << range_high_ ;


    analyze_scaling_factor_gpu(scaling_factor, min_value, max_value, bit_width_, range_low_,range_high_, round_strategy_);
//    analyze_scaling_factor_gpu(scaling_factor, min_value, max_value);

//LOG(INFO) << " scaling_factor" << scaling_factor << " min_value" << min_value << "   max_value" << max_value;


//Dtype *weight = NULL;
int cnt = this->blobs_[0]->count();
//LOG(INFO) << "count====" << this->blobs_[0]->count();

    for (int i = 0; i < (this->blobs_[0]->count()); ++i) 
      {
	fixed_point_gpu( *(Q_weight+i),  round_method_,  scaling_factor, min_value, max_value);
      }
//fixed_point_gpu<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(cnt, Q_weight,  round_method_,  scaling_factor, min_value, max_value, weight);




const Dtype *in_weight = Q_weight;
Dtype *A_weight;
cudaMalloc((void**)&A_weight, this->blobs_[0]->count() * sizeof(Dtype));
caffe_gpu_memcpy(this->blobs_[0]->count() * sizeof(Dtype), in_weight, A_weight);
const Dtype *weight = A_weight;
//LOG(INFO) << A_weight[0] << "++++A_weight" << A_weight[1];




    /**************************************/
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* top_data = top[i]->mutable_gpu_data();
      for (int n = 0; n < this->num_; ++n) {
	this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
			       top_data + n * this->top_dim_);
	if (this->bias_term_) {
	  const Dtype* bias = this->blobs_[1]->gpu_data();
	  this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
	}
      }
    }
  free(tmp_weight);
cudaFree(A_weight);
 }

















  template <typename Dtype>
  void QuanConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* weight = this->blobs_[0]->gpu_data();

    // for (int i = 0; i < 1; ++i) 
    //   cout << "BP--weight" << weight[i] << endl;

    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    for (int i = 0; i < top.size(); ++i) {
      const Dtype* top_diff = top[i]->gpu_diff();
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      // Bias gradient, if necessary.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
	Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
	for (int n = 0; n < this->num_; ++n) {
	  this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
	}
      }
      if (this->param_propagate_down_[0] || propagate_down[i]) {
	for (int n = 0; n < this->num_; ++n) {
	  // gradient w.r.t. weight. Note that we will accumulate diffs.
	  if (this->param_propagate_down_[0]) {
	    this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
				  top_diff + n * this->top_dim_, weight_diff);
	  }
	  // gradient w.r.t. bottom data, if necessary.
	  if (propagate_down[i]) {
	    this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
				    bottom_diff + n * this->bottom_dim_);
	  }
	}
      }
    }
  }


INSTANTIATE_LAYER_GPU_FUNCS(QuanConvolutionLayer);

}  // namespace caffe
