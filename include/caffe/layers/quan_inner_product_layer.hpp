#ifndef CAFFE_QUAN_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_QUAN_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class QuanInnerProductLayer : public Layer<Dtype> {
 public:
  explicit QuanInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "QuanInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights


  /////////////////////////////////////////////////////////////
  int bit_width_;	
  float range_low_, range_high_;
  QuanInnerProductParameter_RoundMethod round_method_;
  QuanInnerProductParameter_RoundStrategy round_strategy_;


  //   void Weight_Quantization(Dtype& weights, float range_low_, float range_high_, int bit_width_ , QuanInnerProductParameter_RoundStrategy round_strategy_, QuanInnerProductParameter_RoundMethod round_method_);
  void Weight_Quantization(Dtype& weights);
  //  void Weight_Quantization(Dtype& weights, float range_low_, float range_high_);
  // void Weight_Quantization(Dtype& weights, float range_low_, float range_high_, int bit_width_ , int* round_strategy_, int* round_method_);

  Dtype fixed_point(const Dtype& input_data, const double& scaling_factor,
		    const double& min_value, const double& max_value) const;

  void analyze_scaling_factor(double& scaling_factor, double& min_value,
			      double& max_value) const;


};

}  // namespace caffe

#endif  // CAFFE_QUAN_INNER_PRODUCT_LAYER_HPP_

