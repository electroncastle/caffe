#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}
#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

bool ReadSegmentRGBToDatum(const string& filename, const int label,
    const vector<int> offsets, const int height, const int width, const int length, Datum* datum, bool is_color)
{
    cv::Mat cv_img;
    string* datum_string;
    char tmp[30];
    int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
        CV_LOAD_IMAGE_GRAYSCALE);
    for (int i = 0; i < offsets.size(); ++i){
        int offset = offsets[i];
        for (int file_id = 1; file_id < length+1; ++file_id){
            sprintf(tmp,"image_%04d.jpg",int(file_id+offset));
            string filename_t = filename + "/" + tmp;
            cv::Mat cv_img_origin = cv::imread(filename_t, cv_read_flag);
            if (!cv_img_origin.data){
                LOG(ERROR) << "Could not load file " << filename;
                return false;
            }
            if (height > 0 && width > 0){
                cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
            }else{
                cv_img = cv_img_origin;
            }
            int num_channels = (is_color ? 3 : 1);
            if (file_id==1 && i==0){
                datum->set_channels(num_channels*length*offsets.size());
                datum->set_height(cv_img.rows);
                datum->set_width(cv_img.cols);
                datum->set_label(label);
                datum->clear_data();
                datum->clear_float_data();
                datum_string = datum->mutable_data();
            }
            if (is_color) {
                for (int c = 0; c < num_channels; ++c) {
                  for (int h = 0; h < cv_img.rows; ++h) {
                    for (int w = 0; w < cv_img.cols; ++w) {
                      datum_string->push_back(
                        static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
                    }
                  }
                }
              } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
                for (int h = 0; h < cv_img.rows; ++h) {
                  for (int w = 0; w < cv_img.cols; ++w) {
                    datum_string->push_back(
                      static_cast<char>(cv_img.at<uchar>(h, w)));
                    }
                  }
              }
        }
    }
    return true;
}

bool ReadSegmentFlowToDatum(const string& filename, const int label,
    const vector<int> offsets, const int height, const int width, const int length, Datum* datum)
{
    cv::Mat cv_img_x, cv_img_y;
    string* datum_string;
    char tmp[30];
    for (int i = 0; i < offsets.size(); ++i){
        int offset = offsets[i];
        for (int file_id = 1; file_id < length+1; ++file_id){
            sprintf(tmp,"flow_x_%04d.jpg",int(file_id+offset));
            string filename_x = filename + "/" + tmp;
            cv::Mat cv_img_origin_x = cv::imread(filename_x, CV_LOAD_IMAGE_GRAYSCALE);
            sprintf(tmp,"flow_y_%04d.jpg",int(file_id+offset));
            string filename_y = filename + "/" + tmp;
            cv::Mat cv_img_origin_y = cv::imread(filename_y, CV_LOAD_IMAGE_GRAYSCALE);
            if (!cv_img_origin_x.data || !cv_img_origin_y.data){
                LOG(ERROR) << "Could not load file " << filename_x << " or " << filename_y;
                return false;
            }
            if (height > 0 && width > 0){
                cv::resize(cv_img_origin_x, cv_img_x, cv::Size(width, height));
                cv::resize(cv_img_origin_y, cv_img_y, cv::Size(width, height));
            }else{
                cv_img_x = cv_img_origin_x;
                cv_img_y = cv_img_origin_y;
            }
            if (file_id==1 && i==0){
                int num_channels = 2;
                datum->set_channels(num_channels*length*offsets.size());
                datum->set_height(cv_img_x.rows);
                datum->set_width(cv_img_x.cols);
                datum->set_label(label);
                datum->clear_data();
                datum->clear_float_data();
                datum_string = datum->mutable_data();
            }
            for (int h = 0; h < cv_img_x.rows; ++h){
                for (int w = 0; w < cv_img_x.cols; ++w){
                    datum_string->push_back(static_cast<char>(cv_img_x.at<uchar>(h,w)));
                }
            }
            for (int h = 0; h < cv_img_y.rows; ++h){
                for (int w = 0; w < cv_img_y.cols; ++w){
                    datum_string->push_back(static_cast<char>(cv_img_y.at<uchar>(h,w)));
                }
            }
        }
    }
    return true;
}

bool ReadOFRToDatum(const string& root_path, const string& img1_file, const string& img2_file,
                    const string& ofx_file, const string& ofy_file,
                    const int height, const int width,
                    Datum* datum, Datum* label_datum, SimpleLMDB *slmdb)
{

    string key = slmdb->get_key(img1_file);
    if (!slmdb->get_Datum(key, label_datum))
        return false;


    cv::Mat img1 = cv::imread(root_path+"/"+img1_file, CV_LOAD_IMAGE_COLOR);
    cv::Mat img2 = cv::imread(root_path+"/"+img2_file, CV_LOAD_IMAGE_COLOR);

    cv::Mat img1_src, img2_src;
    if (!img1.data ){
        LOG(ERROR) << "Could not load file " << img1_file;
        return false;
    }

    if (!img2.data ){
        LOG(ERROR) << "Could not load file " << img2_file;
        return false;
    }

    if (height > 0 && width > 0){
        cv::resize(img1, img1_src, cv::Size(width, height));
        cv::resize(img2, img2_src, cv::Size(width, height));
    }else{
        img1_src = img1;
        img2_src = img2;
    }

    string* datum_string;
    int num_imgs = 2;
    int num_channels = 3;

    datum->set_channels(num_channels*num_imgs);
    datum->set_height(img1.rows);
    datum->set_width(img1.cols);
    datum->set_label(0);
    datum->clear_data();
    datum->clear_float_data();
    datum_string = datum->mutable_data();


    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < img1.rows; ++h) {
        for (int w = 0; w < img1.cols; ++w) {
          datum_string->push_back(
            static_cast<char>(img1.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }

    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < img2.rows; ++h) {
        for (int w = 0; w < img2.cols; ++w) {
          datum_string->push_back(
            static_cast<char>(img2.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }

    return true;

//----------------------------------------------
// Optical Flow
    cv::Mat ofx = cv::imread(root_path+"/"+ofx_file, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat ofy = cv::imread(root_path+"/"+ofy_file, CV_LOAD_IMAGE_GRAYSCALE);

    if (!ofx.data ){
        LOG(ERROR) << "Could not load file " << ofx_file;
        return false;
    }

    if (!ofy.data ){
        LOG(ERROR) << "Could not load file " << ofy_file;
        return false;
    }

    string* label_datum_string;
    num_imgs = 2;
    num_channels = 1;

        label_datum->set_channels(num_channels*num_imgs);
        label_datum->set_height(ofx.rows);
        label_datum->set_width(ofx.cols);
        label_datum->set_label(0);
        label_datum->clear_data();
        label_datum->clear_float_data();
        label_datum_string = label_datum->mutable_data();


//    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < ofx.rows; ++h) {
        for (int w = 0; w < ofx.cols; ++w) {
          label_datum_string->push_back(
            static_cast<char>(ofx.at<char>(h, w)));
        }
      }
//    }

//    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < ofy.rows; ++h) {
        for (int w = 0; w < ofy.cols; ++w) {
          label_datum_string->push_back(
            static_cast<char>(ofy.at<char>(h, w)));
        }
      }
//    }

/*
    cv::Mat cv_img;
    string* datum_string;
    char tmp[30];
    int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
        CV_LOAD_IMAGE_GRAYSCALE);
    for (int i = 0; i < offsets.size(); ++i){
        int offset = offsets[i];
        for (int file_id = 1; file_id < length+1; ++file_id){
            sprintf(tmp,"image_%04d.jpg",int(file_id+offset));
            string filename_t = filename + "/" + tmp;
            cv::Mat cv_img_origin = cv::imread(filename_t, cv_read_flag);
            if (!cv_img_origin.data){
                LOG(ERROR) << "Could not load file " << filename;
                return false;
            }
            if (height > 0 && width > 0){
                cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
            }else{
                cv_img = cv_img_origin;
            }
            int num_channels = (is_color ? 3 : 1);
            if (file_id==1 && i==0){
                datum->set_channels(num_channels*length*offsets.size());
                datum->set_height(cv_img.rows);
                datum->set_width(cv_img.cols);
                datum->set_label(label);
                datum->clear_data();
                datum->clear_float_data();
                datum_string = datum->mutable_data();
            }
            if (is_color) {
                for (int c = 0; c < num_channels; ++c) {
                  for (int h = 0; h < cv_img.rows; ++h) {
                    for (int w = 0; w < cv_img.cols; ++w) {
                      datum_string->push_back(
                        static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
                    }
                  }
                }
              } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
                for (int h = 0; h < cv_img.rows; ++h) {
                  for (int w = 0; w < cv_img.cols; ++w) {
                    datum_string->push_back(
                      static_cast<char>(cv_img.at<uchar>(h, w)));
                    }
                  }
              }
        }
    }
*/
    return true;
}


void showDatum(Datum &datum, int width, int height, int offset)
{
    cv::Mat im(height, width, CV_8UC1);

    int p = offset;
    for (int h = 0; h < im.rows; ++h) {
      for (int w = 0; w < im.cols; ++w) {
          float f = datum.float_data(p++);
          f = f*127+127;
          im.at<uchar>(h, w) = uchar(round(f));
        }
    }

    cv::imshow("label", im);
    cv::waitKey(0);
}

void showXY(Datum &datum, std::string img1_file, std::string img2_file)
{
    int width=32;
    int height=32;

    cv::Mat xim(height, width, CV_8UC1);
    cv::Mat yim(height, width, CV_8UC1);

    int p = 0;
    for (int h = 0; h < xim.rows; ++h) {
      for (int w = 0; w < xim.cols; ++w) {
          float f = datum.float_data(p++);
          f = f*127+127;
          xim.at<uchar>(h, w) = uchar(round(f));
        }
    }

    for (int h = 0; h < xim.rows; ++h) {
      for (int w = 0; w < xim.cols; ++w) {
          float f = datum.float_data(p++);
          f = f*127+127;
          yim.at<uchar>(h, w) = uchar(round(f));
        }
    }

    cv::Mat img1 = cv::imread(img1_file);
    cv::Mat img2 = cv::imread(img2_file);

    cv::Mat xims, yims;
    cv::resize(xim, xims, cv::Size(img1.rows, img1.cols));
    cv::resize(yim, yims, cv::Size(img1.rows, img1.cols));

    cv::cvtColor(xims, xims, CV_GRAY2RGB);
    cv::cvtColor(yims, yims, CV_GRAY2RGB);

    cv::Mat dst, dst1;
    cv::hconcat(img1, img2, dst);
    cv::hconcat(xims, xims, dst1);
    cv::Mat H;
    cv::hconcat(dst, dst1, H);

    cv::imshow("label", H);
    cv::waitKey(2);

}


#endif

}  // namespace caffe
