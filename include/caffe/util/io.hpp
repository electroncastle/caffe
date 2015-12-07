#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <boost/filesystem.hpp>
#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "google/protobuf/message.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

#ifndef CAFFE_TMP_DIR_RETRIES
#define CAFFE_TMP_DIR_RETRIES 100
#endif

#include "lmdb.h"
#include <boost/algorithm/string.hpp>

namespace caffe {

using ::google::protobuf::Message;
using ::boost::filesystem::path;

class SimpleLMDB{
public:
    MDB_txn* mdb_txn;
    MDB_cursor* mdb_cursor;
    MDB_env* mdb_env_;
    MDB_dbi mdb_dbi_;

    void MDB_CHECK(int mdb_status) {
      CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
    }

    void open(const string& source)
    {
      MDB_CHECK(mdb_env_create(&mdb_env_));
      MDB_CHECK(mdb_env_set_mapsize(mdb_env_, 1099511627776));
    //  if (mode == NEW) {
    //    CHECK_EQ(mkdir(source.c_str(), 0744), 0) << "mkdir " << source << "failed";
    //  }
      int flags = 0;
      //if (mode == READ) {
        flags = MDB_RDONLY | MDB_NOTLS;
      //}
      MDB_CHECK(mdb_env_open(mdb_env_, source.c_str(), flags, 0664));
      LOG(INFO) << "Opened lmdb " << source;
    }

    void new_cursor()
    {
      MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn));
      MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_));
      MDB_CHECK(mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor));
      //return new LMDBCursor(mdb_txn, mdb_cursor);
    }


    bool get_val(const string& id, MDB_val &val)
    {
        MDB_val key;
        key.mv_data = strdup(id.c_str()); // strdup("0202000096");
        key.mv_size = strlen((const char*)key.mv_data);

        val.mv_data = 0;
        val.mv_size = 0;
        bool result = mdb_cursor_get(mdb_cursor, &key, &val, MDB_SET) == 0;
        free(key.mv_data);

        return result;
    }

    string valueToString(MDB_val &val)
    {
      return string(static_cast<const char*>(val.mv_data),
          val.mv_size);
    }


    // Try to scale the labels to range (-1..1)
    Datum *valueToDatum(MDB_val &val)
    {
        Datum* datum = new Datum();
        // TODO deserialize in-place instead of copy?
        datum->ParseFromString(valueToString(val));
        return datum;
    }

    Datum *get_Datum(const string& id)
    {
        MDB_val val;
        if (!get_val(id, val))
            return NULL;

        return valueToDatum(val);
    }

    bool get_Datum(const string& id, Datum *datum)
    {
        MDB_val val;
        if (!get_val(id, val))
            return false;
        return datum->ParseFromString(valueToString(val));
    }

    string get_key(const string &filename)
    {
        std::vector<std::string> strs;
        boost::split(strs, filename, boost::is_any_of("/._"));

        int l=strs.size();

        int img_id = atoi(strs[l-2].c_str());
        int vid_id = atoi(strs[l-4].c_str());

        char result[128];
        sprintf(result, "%05d%05d", vid_id, img_id);

        return result;
    }

};


inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  const path& model =
    boost::filesystem::temp_directory_path()/"caffe_test.%%%%-%%%%";
  for ( int i = 0; i < CAFFE_TMP_DIR_RETRIES; i++ ) {
    const path& dir = boost::filesystem::unique_path(model).string();
    bool done = boost::filesystem::create_directory(dir);
    if ( done ) {
      *temp_dirname = dir.string();
      return;
    }
  }
  LOG(FATAL) << "Failed to create a temporary directory.";
}

inline void MakeTempFilename(string* temp_filename) {
  static path temp_files_subpath;
  static uint64_t next_temp_file = 0;
  temp_filename->clear();
  if ( temp_files_subpath.empty() ) {
    string path_string="";
    MakeTempDir(&path_string);
    temp_files_subpath = path_string;
  }
  *temp_filename =
    (temp_files_subpath/caffe::format_int(next_temp_file++, 9)).string();
}

bool ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}


void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool ReadFileToDatum(const string& filename, const int label, Datum* datum);

inline bool ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, -1, datum);
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, is_color,
                          "", datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}

bool DecodeDatumNative(Datum* datum);
bool DecodeDatum(Datum* datum, bool is_color);

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width);

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename);

cv::Mat DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);
#endif  // USE_OPENCV

bool ReadSegmentFlowToDatum(const string& filename, const int label,
    const vector<int> offsets, const int height, const int width, const int length, Datum* datum);

bool ReadSegmentRGBToDatum(const string& filename, const int label,
    const vector<int> offsets, const int height, const int width, const int length, Datum* datum, bool is_color);

bool ReadOFRToDatum(const string& root_path, const string& img1_file, const string& img2_file,
                    const string& ofx_file, const string& ofy_file,
                    const int height, const int width,
                    Datum* datum, Datum* label_datum, SimpleLMDB *slmdb);

void showDatum(Datum &datum, int width, int height, int offset);
void showXY(Datum &datum, std::string img1_file, std::string img2_file);

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
