#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");

  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(), top[0]->mutable_gpu_data());

  int label_offset = 1;
  bool has_data2 =  batch->data2_.shape().size() > 0;
  if (has_data2){
      label_offset = 2;
      top[1]->ReshapeLike(batch->data2_);
      // Copy the data
      caffe_copy(batch->data2_.count(), batch->data2_.gpu_data(), top[1]->mutable_gpu_data());
  }


  if (this->output_labels_) {

     if (batch->blobs_.size() > 0){

         for (int i=0; i<batch->blobs_.size(); i++){
             // Reshape to loaded labels.
           top[i+label_offset]->ReshapeLike(*batch->blobs_[i]);
           // Copy the labels.
           caffe_copy(batch->blobs_[i]->count(), batch->blobs_[i]->gpu_data(),
               top[i+label_offset]->mutable_gpu_data());
         }

     }else{
          // Reshape to loaded labels.
        top[1]->ReshapeLike(batch->label_);
        // Copy the labels.
        caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
            top[1]->mutable_gpu_data());
     }


  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
