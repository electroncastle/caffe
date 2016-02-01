#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/npy_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/cnpy.h"

namespace caffe {

template <typename Dtype>
NPYDataLayer<Dtype>::~NPYDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void NPYDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  const int new_height = this->layer_param_.npy_data_param().new_height();
  const int new_width  = this->layer_param_.npy_data_param().new_width();
  const bool is_color  = this->layer_param_.npy_data_param().is_color();
  string root_folder = this->layer_param_.npy_data_param().root_folder();


  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";


  const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
  frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));

  // Read the file with filenames and labels
  const string& source = this->layer_param_.npy_data_param().source();
  const string& source_path_rgb = this->layer_param_.npy_data_param().source_path_rgb();

  bool rgb_features = false;
  if (source_path_rgb!=""){
      rgb_features = true;
      LOG(INFO) << "********* LOADING RGB feature vectors from "<<source_path_rgb;
  }

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

  if (this->layer_param_.npy_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.npy_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.npy_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // Read an image, and use it to initialize the top blob.
//  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
//                                    new_height, new_width, is_color);

//  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][0];
//  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
//  this->transformed_data_.Reshape(top_shape);

  cnpy::NpyArray npy_arr = cnpy::npy_load(lines_[lines_id_][0]);
  CHECK(npy_arr.data) << "Could not load " << lines_[lines_id_][0];
  int dtype_size = sizeof(Dtype);
  CHECK(sizeof(Dtype) == npy_arr.word_size) << "Data type mismatch. Caffe=" << dtype_size << "  npy data=" << npy_arr.word_size;
  float* npy_arr_data = reinterpret_cast<float*>(npy_arr.data);

  // rows of the npy matrix are LSTM streams
  // columns are date at discrete time steps


  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.npy_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape;
  top_shape.push_back(batch_size);
  top_shape.push_back(1);
  top_shape.push_back(1);
  top_shape.push_back(rgb_features ? npy_arr.shape[1]*2 : npy_arr.shape[1]);

  delete npy_arr_data;

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);


  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();


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
void NPYDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void NPYDataLayer<Dtype>::load_batch(Batch<Dtype>* batch)
{
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  //CHECK(this->transformed_data_.count());
  NPYDataParameter npy_data_param = this->layer_param_.npy_data_param();
  const int batch_size = npy_data_param.batch_size();
  const int new_height = npy_data_param.new_height();
  const int new_width = npy_data_param.new_width();
  const bool is_color = npy_data_param.is_color();
  string root_folder = npy_data_param.root_folder();
  const string& source_path_rgb = this->layer_param_.npy_data_param().source_path_rgb();

  bool rgb_features = false;
  if (source_path_rgb!="")
      rgb_features = true;


  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
//  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
//      new_height, new_width, is_color);
//  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

//  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
//      new_height, new_width, is_color);
//  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_][0];
//  // Use data_transformer to infer the expected blob shape from a cv_img.
//  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
//  this->transformed_data_.Reshape(top_shape);

  cnpy::NpyArray npy_arr = cnpy::npy_load(lines_[lines_id_][0]);
  CHECK(npy_arr.data) << "Could not load " << lines_[lines_id_][0];
  float* npy_arr_data = reinterpret_cast<float*>(npy_arr.data);
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape;
  top_shape.push_back(0);
  top_shape.push_back(1);
  top_shape.push_back(1);
  top_shape.push_back(rgb_features ? npy_arr.shape[1]*2 : npy_arr.shape[1]);
  delete npy_arr_data;

  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_data2 = NULL;
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();


  //int top_size = this->layer_param_.top_size();
  vector<Dtype*> prefetch_blobs;
  prefetch_blobs.resize(batch->blobs_.size());
  for (int l=0; l<batch->blobs_.size(); l++){
    prefetch_blobs[l] = batch->blobs_[l]->mutable_cpu_data();
  }


  caffe::rng_t* rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
  int npy_rows_max = 10;
  int npy_row_rnd = ((*rng)() % npy_rows_max);

  // Fill the batch
  const int lines_size = lines_.size();
  int npy_row = -1; // force to load a new npy array
  int npy_rows = 0;
  npy_arr_data = NULL;
  float* npy_arr_data_rgb =  NULL;

  int data_dim = top_shape[3];
  float *item_data = NULL;
  item_data = new float[data_dim]();
  cnpy::NpyArray npy_arr_rgb;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();


if (lines_[lines_id_][0][0] == '-'){
    for (int i=0; i<data_dim; i++)
        item_data[i] = 0;
}else{
    CHECK_GT(lines_size, lines_id_);
    if (npy_row<0){
        npy_row=0;
        npy_arr = cnpy::npy_load(lines_[lines_id_][0]);
        CHECK(npy_arr.data) << "Could not load " << lines_[lines_id_][0];
        npy_arr_data = reinterpret_cast<float*>(npy_arr.data);
        npy_rows = npy_arr.shape[0];

        if (rgb_features){
            std::string s = lines_[lines_id_][0];
            while (true){
                int fname_begin = s.find_last_of('/');
                if (fname_begin == string::npos)
                    break;

                std::string filename = "fc6_rgb"+s.substr(fname_begin+4);
                s = s.substr(0, fname_begin);
                int path_begin = s.find_last_of('/');

                if (path_begin == string::npos)
                    break;
                s = s.substr(0, path_begin);
                path_begin = s.find_last_of('/');
                if (path_begin == string::npos)
                    break;
                s = lines_[lines_id_][0].substr(path_begin, fname_begin-path_begin);
                std::string path = source_path_rgb+"/"+s+"/"+filename;


                npy_arr_rgb = cnpy::npy_load(path);
                CHECK(npy_arr_rgb.data) << "Could not load " << path;
                npy_arr_data_rgb = reinterpret_cast<float*>(npy_arr_rgb.data);
                //npy_rows = npy_arr_rgb.shape[0];

                break;
            }
        }


    }

    read_time += timer.MicroSeconds();
    timer.Start();


//    float *item_data = &npy_arr_data[npy_row*data_dim];

    // use only use the 10th row
    npy_row = npy_row_rnd;
    if (rgb_features){
        int dim = data_dim/2;
        memcpy(item_data, &npy_arr_data[npy_row*dim], dim*sizeof(float));
        memcpy(item_data+dim, &npy_arr_data_rgb[npy_row*dim], dim*sizeof(float));
        delete npy_arr_data;
        delete npy_arr_data_rgb;
    }else{
        memcpy(item_data, &npy_arr_data[npy_row*data_dim], data_dim);
        delete npy_arr_data;
    }
}

    int offset = batch->data_.offset(item_id);

    // Make sure we read a new file next time
    npy_row = 9;


    caffe_copy(data_dim,
        (Dtype*)item_data,
        &prefetch_data[offset]);

    trans_time += timer.MicroSeconds();


    // Load labels
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

    }
    
    
    // First consume all rows from the npy array
    if (++npy_row < npy_rows){
        continue;
    }else{
        npy_row = -1;
//        if (npy_arr_data){
//            delete npy_arr_data;
//            npy_arr_data = NULL;
//        }else{
//            if (item_data)
//                delete item_data;
//        }
        // Now a new npy array gets loaded
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

  delete item_data;

  if (npy_row>0){
      DLOG(INFO) << "Wrong data alignment. Batch size=" << batch_size << "  npy data rows=" << npy_rows;
  }


  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}



INSTANTIATE_CLASS(NPYDataLayer);
REGISTER_LAYER_CLASS(NPYData);

}  // namespace caffe
#endif  // USE_OPENCV
