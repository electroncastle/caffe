/*
 * Author: Jiri Fajtl <ok1zjf@gmail.com>
 * Date: 20/01/2016
 * Distributed under The MIT License (MIT)
 * Version: 0.1
 *
 *
 */


#ifndef CAFFE_NPY_DATA_LAYER_HPP_
#define CAFFE_NPY_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class NPYDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit NPYDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~NPYDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NPYData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return -1; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  shared_ptr<Caffe::RNG> frame_prefetch_rng_;

  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  //vector<std::pair<std::string, int> > lines_;  
  int lines_id_;

  bool has_two_imgs;
  vector<vector<std::string> > lines_;
//  std::vector<shared_ptr<Blob<Dtype> > > label_blobs_;
  Blob<Dtype> transformed_data2_;
};


}  // namespace caffe

#endif  // CAFFE_NPY_DATA_LAYER_HPP_
