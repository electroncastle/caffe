#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/ofr_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/util/benchmark.hpp"

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif

#include <boost/algorithm/string.hpp>

namespace caffe{
template <typename Dtype>
OFRDataLayer<Dtype>:: ~OFRDataLayer<Dtype>(){
//	this->JoinPrefetchThread();
}


template <typename Dtype>
void OFRDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{          
    label_transformer_.reset(
        new DataTransformer<Dtype>(label_transform_param_, this->phase_));
    label_transformer_->InitRand();

    const int new_height  = this->layer_param_.ofr_data_param().new_height();
    const int new_width  = this->layer_param_.ofr_data_param().new_width();
//    const int new_length  = this->layer_param_.ofr_data_param().new_length();
//    const int num_segments = this->layer_param_.ofr_data_param().num_segments();
    const string& source = this->layer_param_.ofr_data_param().source();
    const string& root_path = this->layer_param_.ofr_data_param().root_path();
    const string& label_source = this->layer_param_.ofr_data_param().label_source();


    slmdb = new SimpleLMDB();
    slmdb->open(label_source);
    slmdb->new_cursor();

//    MDB_val val;
//    lmdb_get_val("0202000096", val);
//    Datum *dat = valueToDatum(val);
    Datum *dat1 = slmdb->get_Datum("0202000096");


	LOG(INFO) << "Opening file: " << source;
	std:: ifstream infile(source.c_str());
    string img1_file;
    string img2_file;
    string ofx_file;
    string ofy_file;
//	int label;
//	int length;

    while (infile >> img1_file >> img2_file >> ofx_file >> ofy_file){
        image_files_.push_back(std::make_pair(img1_file, img2_file));
        of_files_.push_back(std::make_pair(ofx_file, ofy_file));
        //lines_duration_.push_back(length);
	}

//	if (this->layer_param_.video_data_param().shuffle()){
//		const unsigned int prefectch_rng_seed = caffe_rng_rand();
//		prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
//		prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
//		ShuffleVideos();
//	}

    LOG(INFO) << "A total of " << (of_files_.size()>>1) << " Optical flow fields to learn.";
    of_id_ = 0;

	Datum datum;
    Datum label_datum;
    const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
    frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));
    //int average_duration = (int) lines_duration_[lines_id_]/num_segments;

//    vector<int> offsets;
//	for (int i = 0; i < num_segments; ++i){
//		caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
//		int offset = (*frame_rng)() % (average_duration - new_length + 1);
//		offsets.push_back(offset+i*average_duration);
//	}

//	if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW)
//		CHECK(ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum));
//	else
//		CHECK(ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, true));

    CHECK(ReadOFRToDatum(root_path, image_files_[of_id_].first, image_files_[of_id_].second,
         of_files_[of_id_].first, of_files_[of_id_].second,
         new_height, new_width,
         &datum, &label_datum, slmdb));

    // Get Of label for the image
//    string key = slmdb->get_key(image_files_[of_id_].first);
//    slmdb->get_Datum(key, label_datum);

    const int crop_size = this->layer_param_.transform_param().crop_size();
    const int batch_size = this->layer_param_.ofr_data_param().batch_size();
	if (crop_size > 0){
		top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
        //this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
          this->prefetch_[i].data_.Reshape(batch_size, datum.channels(),crop_size, crop_size);
        }

	} else {
		top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
        //this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
          this->prefetch_[i].data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
        }
	}
	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

//	top[1]->Reshape(batch_size, 1, 1, 1);
    //this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
//    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
//      this->prefetch_[i].label_.Reshape(batch_size, 1, 1, 1);
//    }

//    top[1]->Reshape(batch_size, label_datum.channels(), label_datum.width(), label_datum.height());
//    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
//        // TODO: fix the w/h
//        this->prefetch_[i].label_.Reshape(batch_size, label_datum.channels(), label_datum.width(), label_datum.height());
//    }

    int label_crop_size = this->label_transform_param_.crop_size();
    if (label_crop_size > 0){
        top[1]->Reshape(batch_size, label_datum.channels(), label_crop_size, label_crop_size);
        //this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
          this->prefetch_[i].label_.Reshape(batch_size, label_datum.channels(),label_crop_size, label_crop_size);
        }

    } else {
        top[1]->Reshape(batch_size, label_datum.channels(), label_datum.height(), label_datum.width());
        //this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
          this->prefetch_[i].label_.Reshape(batch_size, label_datum.channels(), label_datum.height(), label_datum.width());
        }
    }
    LOG(INFO) << "Label output data size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();


    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);

    // JFMOD
    // the old data transformer doesn't suppoer the infer bolobfcn
    //int crop_size = 224;
//    vector<int> top_shape(4);
//    top_shape[0] = 1;
//    top_shape[1] = datum.channels();
//    top_shape[2] = (crop_size)? crop_size: datum.height();
//    top_shape[3] = (crop_size)? crop_size: datum.width();

    this->transformed_data_.Reshape(top_shape);

    //this->transformed_data_.Reshape(1, datum.channels(), datum.height(), datum.width());
    //this->transformed_label_.Reshape(1, label_datum.channels(), label_datum.height(), label_datum.width());

    vector<int> top_label_shape = this->label_transformer_->InferBlobShape(label_datum);
    this->transformed_label_.Reshape(top_label_shape);

}

template <typename Dtype>
void OFRDataLayer<Dtype>::ShuffleVideos(){
	caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
	caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
	shuffle(lines_duration_.begin(), lines_duration_.end(),prefetch_rng2);
}

/*
template <typename Dtype>
void VideoDataLayer<Dtype>::InternalThreadEntry(){

	Datum datum;
	CHECK(this->prefetch_data_.count());
	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
	VideoDataParameter video_data_param = this->layer_param_.video_data_param();
	const int batch_size = video_data_param.batch_size();
	const int new_height = video_data_param.new_height();
	const int new_width = video_data_param.new_width();
	const int new_length = video_data_param.new_length();
	const int num_segments = video_data_param.num_segments();
	const int lines_size = lines_.size();

	for (int item_id = 0; item_id < batch_size; ++item_id){
		CHECK_GT(lines_size, lines_id_);
		vector<int> offsets;
		int average_duration = (int) lines_duration_[lines_id_] / num_segments;
		for (int i = 0; i < num_segments; ++i){
			if (this->phase_==TRAIN){
				caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
				int offset = (*frame_rng)() % (average_duration - new_length + 1);
				offsets.push_back(offset+i*average_duration);
			} else{
				offsets.push_back(int((average_duration-new_length+1)/2 + i*average_duration));
			}
		}
		if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW){
			if(!ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum)) {
				continue;
			}
		} else{
			if(!ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, true)) {
				continue;
			}
		}
		int offset1 = this->prefetch_data_.offset(item_id);
    	this->transformed_data_.set_cpu_data(top_data + offset1);
		this->data_transformer_->Transform(datum, &(this->transformed_data_));
		top_label[item_id] = lines_[lines_id_].second;
		//LOG()

		//next iteration
		lines_id_++;
		if (lines_id_ >= lines_size) {
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if(this->layer_param_.video_data_param().shuffle()){
				ShuffleVideos();
			}
		}
	}
}
*/

// This function is called on prefetch thread
template<typename Dtype>
void OFRDataLayer<Dtype>::load_batch(Batch<Dtype>* batch)
{
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_imgs_time = 0;
    double read_labels_time = 0;
    double trans_time = 0;
    CPUTimer timer;

    Datum datum;
    Datum label_datum;
    //CHECK(this->prefetch_data_.count());
    CHECK(batch->data_.count());

//    Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
//    Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = batch->label_.mutable_cpu_data();


    OFRDataParameter ofr_data_param = this->layer_param_.ofr_data_param();
    const int batch_size = ofr_data_param.batch_size();
    const int new_height = ofr_data_param.new_height();
    const int new_width = ofr_data_param.new_width();
//    const int new_length = ofr_data_param.new_length();
//    const int num_segments = ofr_data_param.num_segments();
    //const int lines_size = lines_.size();
    const string& root_path = this->layer_param_.ofr_data_param().root_path();

    for (int item_id = 0; item_id < batch_size; ++item_id){
//        CHECK_GT(lines_size, lines_id_);

//        vector<int> offsets;
//        int average_duration = (int) lines_duration_[lines_id_] / num_segments;
//        for (int i = 0; i < num_segments; ++i){
//            if (this->phase_==TRAIN){
//                caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
//                int offset = (*frame_rng)() % (average_duration - new_length + 1);
//                offsets.push_back(offset+i*average_duration);
//            } else{
//                offsets.push_back(int((average_duration-new_length+1)/2 + i*average_duration));
//            }
//        }

//        if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW){
//            if(!ReadSegmentFlowToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum)) {
//                continue;
//            }
//        } else{
//            if(!ReadSegmentRGBToDatum(lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, true)) {
//                continue;
//            }
//        }

        timer.Start();
        CHECK(ReadOFRToDatum(root_path, image_files_[of_id_].first, image_files_[of_id_].second,
             of_files_[of_id_].first, of_files_[of_id_].second,
             new_height, new_width,
             &datum, &label_datum, slmdb));

        read_imgs_time += timer.MicroSeconds();

//        timer.Start();
//        read_labels_time += timer.MicroSeconds();

        //LOG(INFO) << "img1/2  " << image_files_[of_id_].first << "  " << image_files_[of_id_].second;
        //LOG(INFO) << "of key  " << key;
        //showXY(label_datum, root_path+image_files_[of_id_].first, root_path+image_files_[of_id_].second);

        // Set data
//        int offset1 = this->prefetch_data_.offset(item_id);
        timer.Start();
//        const int crop_size = this->layer_param_.transform_param().crop_size();
//        vector<pair<int , int> > offset_pairs;
//        fillFixOffset(datum.height(), datum.width(), crop_size, crop_size, offset_pairs);

        //shared_ptr<Caffe::RNG> rng_;
        caffe::rng_t* rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
        int crop_positions = 13;
        int sel = ((*rng)() % crop_positions);

        bool do_mirror = ((*rng)() % 2) && this->layer_param_.transform_param().mirror();

        int offset1 = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(top_data + offset1);
        this->data_transformer_->Transform(datum, &(this->transformed_data_), sel, do_mirror);

//        Dtype* transformed_data = this->transformed_data_.mutable_cpu_data();
//        this->data_transformer_->Transform(datum, transformed_data);


        // Set label
        int offset2 = batch->label_.offset(item_id);
        this->transformed_label_.set_cpu_data(top_label+offset2);
        this->label_transformer_->Transform(label_datum, &(this->transformed_label_), sel, do_mirror);
        trans_time += timer.MicroSeconds();

//        top_label[item_id] =
//        top_label[item_id] = lines_[lines_id_].second;

        //LOG()

        //next iteration
//        lines_id_++;
//        if (lines_id_ >= lines_size) {
//            DLOG(INFO) << "Restarting data prefetching from start.";
//            lines_id_ = 0;
//            if(this->layer_param_.video_data_param().shuffle()){
//                ShuffleVideos();
//            }
//        }

        of_id_++;
        if (of_id_ >= image_files_.size()){
            if (this->phase_==TRAIN){
                printf("TRAIN: Reseting dataset counter. Max samples: %d\n", image_files_.size());
            }else{
                printf("TEST: Reseting dataset counter. Max samples: %d\n", image_files_.size());
            }
            of_id_ = 0;
        }

    }



//    timer.Stop();
//    batch_timer.Stop();
//    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
//    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
//    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

#ifdef BENCHMARK_DATA
  timer.Stop();
  batch_timer.Stop();
  LOG(INFO) << "  Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "Read images time: " << read_imgs_time / 1000 << " ms.";
  LOG(INFO) << "Read labels time: " << read_labels_time / 1000 << " ms.";
  LOG(INFO) << "  Transform time: " << trans_time / 1000 << " ms.";
#endif

//-----------------------------------------------------------------------------
/*
  CPUTimer batch_timer;
  batch_timer.Start();
  double deque_time = 0;
  double decod_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape on single input batches for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  bool force_color = this->layer_param_.data_param().force_encoded_color();
  if (batch_size == 1 && crop_size == 0) {
    Datum& datum = *(reader_.full().peek());
    if (datum.encoded()) {
      if (force_color) {
        DecodeDatum(&datum, true);
      } else {
        DecodeDatumNative(&datum);
      }
    }
    batch->data_.Reshape(1, datum.channels(),
        datum.height(), datum.width());
    this->transformed_data_.Reshape(1, datum.channels(),
        datum.height(), datum.width());
  }

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    deque_time += timer.MicroSeconds();

    timer.Start();
    cv::Mat cv_img;
    if (datum.encoded()) {
      if (force_color) {
        cv_img = DecodeDatumToCVMat(datum, true);
      } else {
        cv_img = DecodeDatumToCVMatNative(datum);
      }
      if (cv_img.channels() != this->transformed_data_.channels()) {
        LOG(WARNING) << "Your dataset contains encoded images with mixed "
        << "channel sizes. Consider adding a 'force_color' flag to the "
        << "model definition, or rebuild your dataset using "
        << "convert_imageset.";
      }
    }
    decod_time += timer.MicroSeconds();

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    if (datum.encoded()) {
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    } else {
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
    }
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  batch_timer.Stop();

#ifdef BENCHMARK_DATA
  LOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "  Dequeue time: " << deque_time / 1000 << " ms.";
  LOG(INFO) << "   Decode time: " << decod_time / 1000 << " ms.";
  LOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
#endif
*/
}

INSTANTIATE_CLASS(OFRDataLayer);
REGISTER_LAYER_CLASS(OFRData);
}
