#ifndef CAFFE_OFR_DATA_LAYER_HPP_
#define CAFFE_OFR_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {


/**
 * @brief Provides data to the Net from video files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class OFRDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
    explicit OFRDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param),
    label_transform_param_(param.label_transform_param())
    {}
    virtual ~OFRDataLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "OFRData"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
    shared_ptr<Caffe::RNG> prefetch_rng_;
    shared_ptr<Caffe::RNG> prefetch_rng_2_;
    shared_ptr<Caffe::RNG> prefetch_rng_1_;
    shared_ptr<Caffe::RNG> frame_prefetch_rng_;

    virtual void load_batch(Batch<Dtype>* batch);
    virtual void ShuffleVideos();
//    virtual void InternalThreadEntry();

#ifdef USE_MPI
    inline virtual void advance_cursor(){
        lines_id_++;
        if (lines_id_ >= lines_.size()) {
            // We have reached the end. Restart from the first.
            DLOG(INFO) << "Restarting data prefetching from start.";
            lines_id_ = 0;
            if (this->layer_param_.video_data_param().shuffle()) {
                ShuffleVideos();
            }
        }
    }
#endif

    vector<std::pair<std::string, std::string> > image_files_;
    vector<std::pair<std::string, std::string> > of_files_;
    vector<int> lines_duration_;
    vector<std::pair<std::string, int> > lines_;
    //int lines_id_;
    int of_id_;

    TransformationParameter label_transform_param_;
    shared_ptr<DataTransformer<Dtype> > label_transformer_;
    Blob<Dtype> transformed_label_;

    SimpleLMDB *slmdb;
};


}  // namespace caffe

#endif  // CAFFE_OFR_DATA_LAYER_HPP_
