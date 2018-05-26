#ifndef CAFFE_QUAN_CONV_LAYER_HPP_
#define CAFFE_QUAN_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_quan_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class QuanConvolutionLayer : public BaseQuanConvolutionLayer<Dtype> {
 public:

  explicit QuanConvolutionLayer(const LayerParameter& param)
      : BaseQuanConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "QuanConvolutionLayer"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

  /////////////////////////////////////////////////////////////
  int bit_width_;
  float range_low_, range_high_;
  QuanConvolutionParameter_RoundMethod round_method_;
  QuanConvolutionParameter_RoundStrategy round_strategy_;
  bool is_runtime_;
  virtual void get_quantization_paramter();
  void Weight_Quantization(Dtype& weights);
  //  void Weight_Quantization_gpu(Dtype& weights);

};

}  // namespace caffe

#endif  // CAFFE_QUAN_CONV_LAYER_HPP_
