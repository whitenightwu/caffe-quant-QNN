#include <vector>

#include "caffe/layers/conv_layer.hpp"
//#include "caffe/util/qnn.hpp"

#include <iostream>

#include <algorithm>
#include <limits>

namespace caffe {

  template <typename Dtype>
  void Weight_Quantization(Dtype& weights, Dtype range_low_, Dtype range_high_)
  {
    // rounded results of input data
    /*
      double* scaling_factor = 0;
      double* min_value = 0;
      double* max_value = 0;
    */
    Dtype scaling_factor = 0;
    Dtype min_value = 0;
    Dtype max_value = 0;

    /******************************************/
    //Linear_quant
    /*
    double range_low_ = -0.4;
    double range_high_ = 0.4;
    */
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
    Dtype neg_scaling_factor = (range_low_ < 0) ? log2(min_value/range_low_) :
      std::numeric_limits<Dtype>::infinity();
    Dtype pos_scaling_factor = (range_high_ > 0) ? log2(max_value/range_high_) :
      std::numeric_limits<Dtype>::infinity();
    scaling_factor = pow((Dtype)2.0, round(std::min(neg_scaling_factor, pos_scaling_factor)));

    /******************************************/

    Dtype weight_rounded;
    weight_rounded = round(weights * scaling_factor);
 
    // y = clip(x, min, max) / scaling_factor; so y in [min/scaling_factor, max/scaling_factor]
    weights = std::min(std::max((Dtype)weight_rounded, (Dtype)(min_value)), (Dtype)(max_value)) /
      (Dtype)(scaling_factor);
  }


  //origin conv_layer below
template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

  template <typename Dtype>
  void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
					    const vector<Blob<Dtype>*>& top) {

    /**************************************/
    Dtype* tmp_weight = (Dtype*) malloc((this->blobs_[0]->count())*sizeof(Dtype));
    caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(), tmp_weight);

    Dtype* sort_weight = tmp_weight;
    int qcount_ = this->blobs_[0]->count();
    std::sort(sort_weight, sort_weight+(this->blobs_[0]->count()));
    Dtype range_high_ = sort_weight[qcount_-1];
    Dtype range_low_ = sort_weight[0];
    /*
    for (int i = 0; i < 1; ++i)
      { 
	std::cout << "old--cpu_data" << this->blobs_[0]->cpu_data()[i] << std::endl;
	//	std::cout << "tmp_weight" << tmp_weight[i] << std::endl;
      }
    */

    Dtype* Q_weight = const_cast<Dtype*>(tmp_weight);
    for (int i = 0; i < (this->blobs_[0]->count()); ++i) 
      {
	Weight_Quantization(*Q_weight, range_low_, range_high_);
      }
    const Dtype *weight = Q_weight;
    for (int i = 0; i < 1; ++i) 
      {
	std::cout << "max:" << range_high_ << "  " << "min:" << range_low_ << std::endl;
	std::cout << "new--cpu_data" << this->blobs_[0]->cpu_data()[i] << std::endl;
      }
 
    /**************************************/
    // const Dtype* weight = this->blobs_[0]->cpu_data();

    //print weight to scence
    for (int i = 0; i < 1; ++i) 
      std::cout << "comput--weight" << weight[i] << std::endl;

    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data();
      for (int n = 0; n < this->num_; ++n) {
	this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
			       top_data + n * this->top_dim_);
	if (this->bias_term_) {
	  const Dtype* bias = this->blobs_[1]->cpu_data();
	  this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
	}
      }
    }

  }

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();

  for (int i = 0; i < 1; ++i) 
    std::cout << "BP--weight" << weight[i] << std::endl;

  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
