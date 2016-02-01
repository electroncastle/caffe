#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_lstm_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageLSTMDataLayer<Dtype>::~ImageLSTMDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageLSTMDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";


  const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
  frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
//  string filename;
//  int label;
//  while (infile >> filename >> label) {
//    lines_.push_back(std::make_pair(filename, label));
//  }
	
	string line;
    int labels_count=0;
    while (std::getline(infile, line)){
		std::stringstream ss(line);
		std::istream_iterator<std::string> begin(ss);
		std::istream_iterator<std::string> end;
		std::vector<std::string> vstrings(begin, end);
    //	std::copy(vstrings.begin(), vstrings.end(), std::ostream_iterator<std::string>(std::cout, "\n"));

        lines_.push_back(vstrings);
    }

    has_two_imgs=false;
    if (lines_.size()>0){
        labels_count = lines_[0].size()-1;
        if (labels_count>2){
            has_two_imgs = true;
            labels_count -= 1;
        }
    }

    int top_size = this->layer_param_.top_size();
    if (top_size-1 < labels_count)
        labels_count = top_size-1;


//    label_blobs_.resize(top_size);

//    for (int i = 0; i < top_size; ++i) {
//      label_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
//      // load labels
//      //hdf_blobs_[i].get()
//    }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
//  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    //new_height, new_width, is_color);
  //CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][0];

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);

  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);


  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();


  if (has_two_imgs){
      cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][1],
                                        new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][1];
      // Use data_transformer to infer the expected blob shape from a cv_image.
      vector<int> top_shape2 = this->data_transformer_->InferBlobShape(cv_img);
      CHECK(!(top_shape[1] != top_shape2[1] || top_shape[2] != top_shape2[2] || top_shape[3] != top_shape2[3])) << "Second image has different resolution than the first one" << lines_[lines_id_][1];
      this->transformed_data2_.Reshape(top_shape2);
      top_shape2[0] = batch_size;
      for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
          this->prefetch_[i].data2_.Reshape(top_shape2);
      }

      top[1]->Reshape(top_shape2);

      LOG(INFO) << "output data size (SECOND IMAGE): " << top[1]->num() << ","
          << top[1]->channels() << "," << top[1]->height() << ","
          << top[1]->width();
  }



  // labels
  //---------------------
  vector<int> label_shape(1, batch_size);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);

      this->prefetch_[i].blobs_.resize(labels_count);
      for (int l=0; l<labels_count; l++){
        this->prefetch_[i].blobs_[l] = new Blob<Dtype>();
      }
  }

  int label_offset = has_two_imgs ? 2 : 1;
  for (int l=0; l<labels_count; l++){
      top[l+label_offset]->Reshape(label_shape);
      for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
          this->prefetch_[i].blobs_[l]->Reshape(label_shape);
      }
  }


}

template <typename Dtype>
void ImageLSTMDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageLSTMDataLayer<Dtype>::load_batch(Batch<Dtype>* batch)
{
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
//  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
//      new_height, new_width, is_color);
//  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][0];
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);

  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_data2 = NULL;
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  cv::Mat cv_img2;
  if (has_two_imgs){
      cv_img2 = ReadImageToCVMat(root_folder + lines_[lines_id_][1],
          new_height, new_width, is_color);
      CHECK(cv_img2.data) << "Could not load " << lines_[lines_id_][1];

      vector<int> top_shape2 = this->data_transformer_->InferBlobShape(cv_img2);
      CHECK(!(top_shape[1] != top_shape2[1] || top_shape[2] != top_shape2[2] || top_shape[3] != top_shape2[3])) << "Second image has different resolution than the first one" << lines_[lines_id_][1];
      this->transformed_data2_.Reshape(top_shape2);
      top_shape2[0] = batch_size;
      batch->data2_.Reshape(top_shape2);
      prefetch_data2 = batch->data2_.mutable_cpu_data();
  }


  //int top_size = this->layer_param_.top_size();
  vector<Dtype*> prefetch_blobs;
  prefetch_blobs.resize(batch->blobs_.size());
  for (int l=0; l<batch->blobs_.size(); l++){
    prefetch_blobs[l] = batch->blobs_[l]->mutable_cpu_data();
  }

  // datum scales

  // Get image transformation parameters here and
  // apply them for the whole batch
  caffe::rng_t* rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
  int crop_positions = 13;
  //int crop_positions = 5; // only basic transformations
  int crop_id = ((*rng)() % crop_positions);
  bool do_mirror = ((*rng)() % 2) && this->layer_param_.transform_param().mirror();

  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
//    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
//        new_height, new_width, is_color);
//    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][0];

    read_time += timer.MicroSeconds();
    timer.Start();

    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    //this->data_transformer_->Transform(cv_img, &(this->transformed_data_));

//    int offset1 = batch->data_.offset(item_id);
//    this->transformed_data_.set_cpu_data(prefetch_data + offset1);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_), crop_id, do_mirror);

    if (has_two_imgs){
        int offset = batch->data2_.offset(item_id);
        cv::Mat cv_img2 = ReadImageToCVMat(root_folder + lines_[lines_id_][1],
            new_height, new_width, is_color);
        CHECK(cv_img2.data) << "Could not load " << lines_[lines_id_][1];
        this->transformed_data2_.set_cpu_data(prefetch_data2 + offset);
        this->data_transformer_->Transform(cv_img2, &(this->transformed_data2_), crop_id, do_mirror);
    }

    trans_time += timer.MicroSeconds();

    int label_id = 0;
    int label_offset = has_two_imgs ? 2 : 1;
    for (int i=label_offset; i<lines_[lines_id_].size(); i++){
//        prefetch_label[item_id] = lines_[lines_id_].second;
//        prefetch_label[item_id] = lines_[lines_id_][i];

//        int offset = batch->blobs_[i]->offset(item_id);
        //this->transformed_data_.set_cpu_data(prefetch_blobs[i] + offset);
        string label_str = lines_[lines_id_][i];
        float label = atof(label_str.c_str());

        if (label_id < prefetch_blobs.size())
            prefetch_blobs[label_id++][item_id] = label;

//xx        this->data_transformer_->Transform(cv_img, &(this->transformed_data_));

    }
    
    
    
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      LOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}



INSTANTIATE_CLASS(ImageLSTMDataLayer);
REGISTER_LAYER_CLASS(ImageLSTMData);

}  // namespace caffe
#endif  // USE_OPENCV
