//
// Created by Administrator on 2019-10-16.
//
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include <jni.h>
#include <iostream>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <cstring>
#include <string>
#include <string.h>
#include <pthread.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>

using namespace cv;

#define  LOG_TAG    "native-dev"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
using namespace std;
//#ifndef _Included_cn_scut_dongxia_opencvdemo_MainActivity
//#define _Included_cn_scut_dongxia_opencvdemo_MainActivity
//#ifndef _Included_com_padyun_lxh_jxqy_MainActivity
//#define _Included_com_padyun_lxh_jxqy_MainActivity
//#ifndef _Included_com_padyun_lxh_jxqy_Jxqy
//#define _Included_com_padyun_lxh_jxqy_Jxqy


#ifdef __cplusplus
extern "C" {
#endif


JNIEXPORT jfloatArray JNICALL
Java_com_padyun_lxh_jxqy_PublicFunction_FindPic(JNIEnv *, jobject, jint, jint, jint, jint,
                                                jbyteArray, jobject, jstring);
JNIEXPORT jfloatArray JNICALL
Java_com_padyun_lxh_jxqy_PublicFunction_FindPicMat(JNIEnv *, jobject, jint, jint, jint, jint, jlong,
                                                   jobject);
JNIEXPORT void JNICALL
Java_com_padyun_publicfunctionlibrary_FunctionClass_multipointFindColor(JNIEnv *env, jobject obj,
                                                                        jint, jint,
                                                                        jint, jint, jbyteArray,
                                                                        jlong, jstring, jstring,
                                                                        jdouble,
                                                                        jintArray);
JNIEXPORT void JNICALL
Java_com_padyun_publicfunctionlibrary_FunctionClass_multipointFindColorEx(JNIEnv *env, jobject obj,
                                                                          jint, jint,
                                                                          jint, jint, jbyteArray,
                                                                          jlong, jstring, jstring,
                                                                          jdouble,
                                                                          jintArray);
JNIEXPORT void JNICALL
Java_com_padyun_publicfunctionlibrary_FunctionClass_test();
JNIEXPORT void JNICALL
Java_com_padyun_lxh_jxqy_PublicFunction_ScreenCap(JNIEnv *env, jobject obj);

mutex my_lock;
bool check_sub;
void testthread(int i, Mat mGr, uchar *d, int row, int col, int *str_start_rgb,
                vector<vector<int> > str_sub_rgb, int rect_w, int rect_y);
void similar_arithmetic(Mat mGr, uchar *d, int row, int col, int *str_start_rgb,
                        vector<vector<int> > str_sub_rgb, vector<vector<int> > str_sub_xy,
                        int rect_w, int rect_y, int *range_, double sim, int *return_value);
void similar_arithmetic_1(Mat mGr, uchar *d, int row, int col, int *str_start_rgb,
                          vector<vector<int> > str_sub_rgb, vector<vector<int> > str_sub_xy,
                          int rect_w, int rect_y, int *range_, double sim, int *return_value);
void sub_similar_arithmetic(int i, int j, vector<vector<int> > str_sub_xy,
                            vector<vector<int> > str_sub_rgb, int rect_y, int rect_w, Mat mGr,
                            int threshold, bool *check, int k);
void calculateSimilarity(Vec3b rgb1, int *rgb2);
vector<string> split(const string &, const string &);
#ifdef __cplusplus
}
#endif


JNIEXPORT jfloatArray JNICALL
Java_com_padyun_lxh_jxqy_PublicFunction_FindPicMat(JNIEnv *env, jobject obj, jint x1, jint y1,
                                                   jint x2, jint y2, jlong imgMat, jobject AM) {
    LOGI("======FindPicMat====================================================================");
    AAssetManager *mgr = AAssetManager_fromJava(env, AM);

    AAsset *asset = AAssetManager_open(mgr, "activity.png", AASSET_MODE_UNKNOWN);

    size_t fileLength = AAsset_getLength(asset);
    char *dataBuffer2 = (char *) malloc(fileLength);
    AAsset_read(asset, dataBuffer2, fileLength);
    AAsset_close(asset);
    std::vector<char> vec2(dataBuffer2, dataBuffer2 + fileLength);
    cv::Mat img1 = cv::imdecode(vec2, CV_LOAD_IMAGE_COLOR);

    cv::Mat mGr = (*((cv::Mat *) imgMat));

    LOGI("======FindPicMat====================================================================%d",
         mGr.channels());
    cv::Mat dstImg;


    dstImg.create(mGr.dims, mGr.size, mGr.type());
    cv::matchTemplate(img1, mGr, dstImg, cv::TM_CCOEFF_NORMED);
    cv::Point minPoint;
    cv::Point maxPoint;
    double minVal = 0;
    double maxVal = 0;
    cv::minMaxLoc(dstImg, &minVal, &maxVal, &minPoint, &maxPoint);

    jfloat *value = new jfloat[3];
    value[0] = (jfloat) 0.1;
    value[1] = (jfloat) 0.1;
    value[2] = (jfloat) 0.1;

    jfloatArray jarr = env->NewFloatArray(3);

    env->SetFloatArrayRegion(jarr, 0, 3, value);

    return jarr;

}


JNIEXPORT jfloatArray JNICALL
Java_com_padyun_lxh_jxqy_PublicFunction_FindPic(JNIEnv *env, jobject obj, jint x1, jint y1, jint x2,
                                                jint y2, jbyteArray imgBytes, jobject AM,
                                                jstring fileName) {

    jbyte *img = env->GetByteArrayElements(imgBytes, NULL);
    cv::Mat imgMat(720, 1280, CV_8UC4, (unsigned char *) img);
    cv::cvtColor(imgMat, imgMat, cv::COLOR_BGRA2RGB);

    cv::Mat rectMat(imgMat, CvRect(x1, y1, x2 - x1, y2 - y1));

    cv::Mat dstImg;
    dstImg.create(rectMat.dims, rectMat.size, rectMat.type());

    AAssetManager *mgr = AAssetManager_fromJava(env, AM);

    const char *str = env->GetStringUTFChars(fileName, JNI_FALSE);
    string s(str);
    vector<string> AllStr;
    if (s.find("|") != string::npos) {
        AllStr = split(str, "|");
    } else {
        AllStr.push_back(s);
    }
    jfloat *value = new jfloat[3];
    jfloat minSim = 0.0;
    for (int i = 0; i < AllStr.size(); ++i) {
        AAsset *asset = AAssetManager_open(mgr, AllStr[i].c_str(), AASSET_MODE_UNKNOWN);
        size_t fileLength = AAsset_getLength(asset);
        char *dataBuffer2 = (char *) malloc(fileLength);
        AAsset_read(asset, dataBuffer2, fileLength);
        AAsset_close(asset);
        std::vector<char> vec2(dataBuffer2, dataBuffer2 + fileLength);
        Mat img1 = cv::imdecode(vec2, CV_LOAD_IMAGE_COLOR);
//        UMat imgU,rectMatU,UdstImg;
//        imgU.copyTo(img1);
//        rectMatU.copyTo(rectMat);
//        UdstImg.copyTo(dstImg);
        LOGI("======FindPicMat====================================================================match");
        cv::matchTemplate(img1, rectMat, dstImg, cv::TM_CCOEFF_NORMED);
        cv::Point minPoint;
        cv::Point maxPoint;
        double minVal = 0;
        double maxVal = 0;
        cv::minMaxLoc(dstImg, &minVal, &maxVal, &minPoint, &maxPoint);
//        LOGI("======FindPicMat====================================================================%lf" ,maxVal);
        if ((jfloat) maxVal > minSim) {
            minSim = (jfloat) maxVal;
            value[0] = (jfloat) maxVal;
            value[1] = (jfloat) maxPoint.x + x1;
            value[2] = (jfloat) maxPoint.y + y1;
        }
        img1.release();
    }


    jfloatArray jarr = env->NewFloatArray(3);
    env->SetFloatArrayRegion(jarr, 0, 3, value);

    free(value);
    rectMat.release();

    imgMat.release();
    dstImg.release();
//    LOGI("======FindPic=============return" );
    return jarr;

}


vector<string> split(const string &str, const string &delim) {
    vector<string> res;
    if ("" == str) return res;
    //先将要切割的字符串从string类型转换为char*类型
    char *strs = new char[str.length() + 1]; //不要忘了
    strcpy(strs, str.c_str());

    char *d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d);
    while (p) {
        string s = p; //分割得到的字符串转换为string类型
        res.push_back(s); //存入结果数组
        p = strtok(NULL, d);
    }

    return res;
}

void calculateSimilarity(Vec3b rgb1, int *rgb2) {
    double trageSim = 0.9;
    double c = 25.5;
    int c1 = rgb1[0] - rgb2[0];
//    if (c1>c){
//        return;
//    }
//    int c2=abs(rgb1[1]-rgb2[1]);
//    if(c2>c){
//        return;
//    }
//    int c3=abs(rgb1[2]-rgb2[2]);
//    if(c3>c){
//        return;
//    }



//    double sim__ =1-(c1+c2+c3)/765;
//    double sim__ =765-(c1+c2+c3);

/*
    int rgb1_rgb2_1=abs(rgb1[0]-rgb2[0]);
    if (rgb1_rgb2_1>c){
        return;
    }

//    double sim1=fabs((rgb1[0]/255) - (rgb2[0]/255));

    double sim1=rgb1_rgb2_1/255;
//    if (sim1/3>c){
//        return;
//    }
//    double sim2=fabs((rgb1[1]/255) - (rgb2[1]/255));

    int rgb1_rgb2_2=abs(rgb1[1]-rgb2[1]);
//    if (rgb1_rgb2_2>c){
//        return;
//    }
    double sim2=rgb1_rgb2_2/255;

//    if((sim1+sim2)/3>c){
//        return;
//    }
//    double sim3=fabs((rgb1[2]/255) - (rgb2[2]/255));

    double sim3=fabs((rgb1[2]-rgb2[2])/255);
    double sim = 1-((sim1+sim2+sim3)/3);

//    LOGI("======multipointFindColor====================================================================%d%d%d",rgb2[0],rgb2[1],rgb2[2] );
//    LOGI("======multipointFindColor====================================================================%d%d%d" ,rgb1[0],rgb1[1],rgb1[2]);
//    double sim =0.0;
//    for (int i = 0; i <3 ; ++i) {
////      sim=sim+pow(pow((int)rgb1[i]/255 -  (int)rgb2[i]/255,2),0.5);
//        sim=sim+fabs((rgb1[i]/255) - (rgb2[i]/255));
//    }

//    if (sim>trageSim){
//        LOGI("======multipointFindColor===================================================================sim=%lf" ,sim);
//        LOGI("======multipointFindColor===================================================================sim1=%lf" ,sim1);
//        LOGI("======multipointFindColor===================================================================sim2=%lf" ,sim2);
//        LOGI("======multipointFindColor===================================================================sim3=%lf" ,sim3);
//    }
*/
}

int myAdd(int num1, int num2) {
    if (num2 == 0) return num1;
    int sum = 0, carry = 0;
    sum = num1 ^ num2;    // 按位抑或
    carry = (num1 & num2) << 1;
    return myAdd(sum, carry);
}


int myMinus(int num1, int num2) {
    return myAdd(num1, myAdd(~num2, 1));
}


//JNIEXPORT void JNICALL Java_com_padyun_lxh_jxqy_MainActivity_color__(JNIEnv *env, jobject obj){
JNIEXPORT void JNICALL
Java_com_padyun_publicfunctionlibrary_FunctionClass_multipointFindColor(JNIEnv *env, jobject obj,
                                                                        jint x_1,
                                                                        jint y_1, jint x_2,
                                                                        jint y_2,
                                                                        jbyteArray imgBytes,
                                                                        jlong imgMat,
                                                                        jstring colorStr_start,
                                                                        jstring colorStr_sub,
                                                                        jdouble targetSim,
                                                                        jintArray xy) {


    int return_value[2] = {-1, -1};

    const char *str_start = env->GetStringUTFChars(colorStr_start, JNI_FALSE);
    const char *str_sub = env->GetStringUTFChars(colorStr_sub, JNI_FALSE);
    int rect_w = x_2 - x_1;
    int rect_y = y_2 - y_1;
    cv::Mat allMat = (*((cv::Mat *) imgMat));
    cv::Mat mGr(allMat, CvRect(x_1, y_1, rect_w, rect_y));

    vector<string> str_start_arr;
    int *str_start_rgb = new int[3];
    str_start_arr = split(str_start, ",");
    str_start_rgb[0] = stoi(str_start_arr[0]);
    str_start_rgb[1] = stoi(str_start_arr[1]);
    str_start_rgb[2] = stoi(str_start_arr[2]);
    vector<string> str_sub_arr;
    vector<string> str_sub_xy_rgb;
    char *str_sub_arr_sub;
    str_sub_arr = split(str_sub, "&");
    int str_sub_rgb[str_sub_arr.size()][3];
    int str_sub_xy[str_sub_arr.size()][2];
    vector<string> str_sub_rgb_all;
    for (int i = 0; i < str_sub_arr.size(); ++i) {
        str_sub_arr_sub = (char *) str_sub_arr[i].c_str();
        str_sub_xy_rgb = split(str_sub_arr_sub, "|");
        str_sub_xy[i][0] = stoi(str_sub_xy_rgb[0]);
        str_sub_xy[i][1] = stoi(str_sub_xy_rgb[1]);
        str_sub_rgb_all = split(str_sub_xy_rgb[2], ",");
        str_sub_rgb[i][0] = stoi(str_sub_rgb_all[0]);
        str_sub_rgb[i][1] = stoi(str_sub_rgb_all[1]);
        str_sub_rgb[i][2] = stoi(str_sub_rgb_all[2]);
    }
    double target = targetSim;
    int c1, c2, c3, c4;
    double sim;

    uchar *d;
    int index;

//    d = mGr.ptr<uchar>(610);
//    int xx=1182;
//    LOGI("======multipointFindColor==========================1=============================%d ",d[xx *3]);
//    LOGI("======multipointFindColor==========================2=============================%d ",d[xx *3+1]);
//    LOGI("======multipointFindColor==========================3=============================%d ",d[xx *3+2]);
//    d = mGr.ptr<uchar>(609);
//    xx=1181;
//    LOGI("======multipointFindColor==========================1=============================%d ",d[xx *3]);
//    LOGI("======multipointFindColor==========================1=============================%d ",d[xx *3+1]);
//    LOGI("======multipointFindColor==========================1=============================%d ",d[xx *3+2]);

    for (int i = 0; i < mGr.rows; ++i) {
        d = mGr.ptr<uchar>(i);
        for (int j = 0; j < mGr.cols; ++j) {
            index = j * 3;
            c1 = d[index] - str_start_rgb[0];
            c1 = (c1 ^ (c1 >> 31)) - (c1 >> 31);
            c2 = d[index + 1] - str_start_rgb[1];
            c2 = (c2 ^ (c2 >> 31)) - (c2 >> 31);
            c3 = d[index + 2] - str_start_rgb[2];
            c3 = (c3 ^ (c3 >> 31)) - (c3 >> 31);
            c4 = 765 - c1 - c2 - c3;
            sim = c4 / 765.0;
            if (sim > target) {
//                LOGI("======multipointFindColor==========================1=============================sim=%lf %d %d ",sim, j,i);
                int x = j;
                int y = i;
                bool check = true;
                for (size_t k = 0; k < sizeof(str_sub_xy) / sizeof(*str_sub_xy); k++) {
                    int row = i + str_sub_xy[k][1];
                    int col = j + str_sub_xy[k][0];
//                    LOGI("======multipointFindColor==========================1==============================%d %d ",row,col);
                    if (row < 0 || row > rect_y || col < 0 || col > rect_w) {
                        check = false;
                        break;
                    }
                    index = col * 3;
                    int r1 = mGr.ptr<uchar>(row)[index];
                    int g1 = mGr.ptr<uchar>(row)[index + 1];
                    int b1 = mGr.ptr<uchar>(row)[index + 2];
                    c1 = r1 - str_sub_rgb[k][0];
                    c1 = (c1 ^ (c1 >> 31)) - (c1 >> 31);
                    c2 = g1 - str_sub_rgb[k][1];
                    c2 = (c2 ^ (c2 >> 31)) - (c2 >> 31);
                    c3 = b1 - str_sub_rgb[k][2];
                    c3 = (c3 ^ (c3 >> 31)) - (c3 >> 31);
                    c4 = c1 + c2 + c3;
                    sim = (765 - c4) / 765.0;
//                   LOGI("======multipointFindColor==========================1==============================%d %d %d %lf",r1,b1,g1,sim);
                    if (sim < target) {
                        check = false;
                        break;
                    }
                }
//                LOGI("======multipointFindColor==========================1=============================sim=%lf ",sim);
                if (check) {

                    delete[] str_start_rgb;
                    str_start_arr.clear();
                    mGr.release();
                    str_sub_arr.clear();
                    str_sub_xy_rgb.clear();
                    str_sub_rgb_all.clear();
                    return_value[0] = x + x_1;
                    return_value[1] = y + y_1;
                    env->SetIntArrayRegion(xy, 0, 2, return_value);
                    return;
                }
            }
        }
    }


    delete[] str_start_rgb;
    str_start_arr.clear();
    mGr.release();
    str_sub_arr.clear();
    str_sub_xy_rgb.clear();
    str_sub_rgb_all.clear();
    env->SetIntArrayRegion(xy, 0, 2, return_value);
//    LOGI("======multipointFindColor==========================return============================= ");
    return;
}

JNIEXPORT void JNICALL
Java_com_padyun_publicfunctionlibrary_FunctionClass_multipointFindColorEx(JNIEnv *env, jobject obj,
                                                                          jint x_1,
                                                                          jint y_1, jint x_2,
                                                                          jint y_2,
                                                                          jbyteArray imgBytes,
                                                                          jlong imgMat,
                                                                          jstring colorStr_start,
                                                                          jstring colorStr_sub,
                                                                          jdouble targetSim,
                                                                          jintArray xy) {


    int return_value[2] = {-1, -1};
    const char *str_start = env->GetStringUTFChars(colorStr_start, JNI_FALSE);
    const char *str_sub = env->GetStringUTFChars(colorStr_sub, JNI_FALSE);
    int rect_w = x_2 - x_1;
    int rect_h = y_2 - y_1;
    cv::Mat allMat = (*((cv::Mat *) imgMat));
    cv::Mat mGr(allMat, CvRect(x_1, y_1, rect_w, rect_h));

    vector<string> str_start_arr;
    int *str_start_rgb = new int[3];
    str_start_arr = split(str_start, ",");
    str_start_rgb[0] = stoi(str_start_arr[0]);
    str_start_rgb[1] = stoi(str_start_arr[1]);
    str_start_rgb[2] = stoi(str_start_arr[2]);
    vector<string> str_sub_arr;
    vector<string> str_sub_xy_rgb;
    char *str_sub_arr_sub;
    str_sub_arr = split(str_sub, "&");
//    int str_sub_rgb[str_sub_arr.size()][3];
//    int *str_sub_rgb[3];
    vector<vector<int> > str_sub_rgb;
    str_sub_rgb.resize(str_sub_arr.size());
//    int str_sub_xy[str_sub_arr.size()][2];
    vector<vector<int> > str_sub_xy;
    str_sub_xy.resize(str_sub_arr.size());
    vector<string> str_sub_rgb_all;
    for (int i = 0; i < str_sub_arr.size(); ++i) {
        str_sub_arr_sub = (char *) str_sub_arr[i].c_str();
        str_sub_xy_rgb = split(str_sub_arr_sub, "|");
        str_sub_xy[i].resize(2);
        str_sub_xy[i][0] = stoi(str_sub_xy_rgb[0]);
        str_sub_xy[i][1] = stoi(str_sub_xy_rgb[1]);
        str_sub_rgb_all = split(str_sub_xy_rgb[2], ",");
        str_sub_rgb[i].resize(3);
        str_sub_rgb[i][0] = stoi(str_sub_rgb_all[0]);
        str_sub_rgb[i][1] = stoi(str_sub_rgb_all[1]);
        str_sub_rgb[i][2] = stoi(str_sub_rgb_all[2]);
    }

    double target = targetSim;
    int c1, c2, c3, c4;
    double sim;

    uchar *d;
    int index;

//    d = mGr.ptr<uchar>(610);
//    int xx=1182;
//    LOGI("======multipointFindColor==========================1=============================%d ",d[xx *3]);
//    LOGI("======multipointFindColor==========================2=============================%d ",d[xx *3+1]);
//    LOGI("======multipointFindColor==========================3=============================%d ",d[xx *3+2]);
//    d = mGr.ptr<uchar>(609);
//    xx=1181;
//    LOGI("======multipointFindColor==========================1=============================%d ",d[xx *3]);
//    LOGI("======multipointFindColor==========================1=============================%d ",d[xx *3+1]);
//    LOGI("======multipointFindColor==========================1=============================%d ",d[xx *3+2]);

    int index_thread = 10;
    int range_[index_thread][4];
    int step_w = rect_w / index_thread, step_h = rect_h / index_thread;
    int constant_=rect_w % index_thread;
    if (constant_>0){
        constant_=1;
    }

    thread threads[index_thread];

    for (int i = 0; i < index_thread; i++) {
        range_[i][0] = i * step_w;
        range_[i][1] = 0;
        if (i+1==index_thread){
            range_[i][2] = rect_w;
        } else{
            range_[i][2] = (i + 1) * step_w + 1;
        }
        range_[i][3] = rect_h;
        threads[i] = thread(similar_arithmetic_1, mGr, d, 1280, 720, str_start_rgb, str_sub_rgb,
                            str_sub_xy, rect_w, rect_h, range_[i], target, (int *) &return_value);
    }
    for (int i = 0; i < index_thread; i++) {
        threads[i].join();
    }
    /*
    int threshold = (int) (765 * (1 - target));

    LOGI("======multipointFindColor==========================threshold= %d ", threshold);
    for (int i = 0; i < rect_h; ++i) {
        d = mGr.ptr<uchar>(i);
        for (int j = 0; j < rect_w; ++j) {
            index = j * 3;
            c1 = d[index] - str_start_rgb[0];
            c1 = (c1 ^ (c1 >> 31)) - (c1 >> 31);
            if (c1 > threshold) continue;
            c2 = d[index + 1] - str_start_rgb[1];
            c2 = (c2 ^ (c2 >> 31)) - (c2 >> 31);
            c3 = d[index + 2] - str_start_rgb[2];
            c3 = (c3 ^ (c3 >> 31)) - (c3 >> 31);//绝对值
            c4 = c1 + c2 + c3;
            if (c4 < threshold) {
//                LOGI("======multipointFindColor==========================1=============================j ,i = %d %d ", j,i);
                int x = j;
                int y = i;
                bool check = true;
                thread threads[str_sub_xy.size()];
                for (size_t k = 0; k < str_sub_xy.size(); k++) {
                    threads[k] = thread(sub_similar_arithmetic, i, j, str_sub_xy, str_sub_rgb,
                                        rect_h, rect_w, mGr, threshold, &check, k);
                }
                for (size_t k = 0; k < str_sub_xy.size(); k++) {
//                    LOGI("======multipointFindColor==========================threads  join============================== ");
                    threads[k].join();
                }
                if (check) {
                    LOGI("======multipointFindColor==========================return==============================%lf ;%d ;%d ",
                         sim, x, y);
                    return;
                }
            }
        }
    }
     */
    if (return_value[0]>-1){

        return_value[0]=return_value[0]+x_1;

        return_value[1]=return_value[1]+y_1;

    }



    delete[] str_start_rgb;
    str_start_arr.clear();
    mGr.release();
    str_sub_arr.clear();
    str_sub_xy_rgb.clear();
    str_sub_rgb_all.clear();
    env->SetIntArrayRegion(xy, 0, 2, return_value);
//    LOGI("======multipointFindColor==========================return============================= ");
    return;
}

void testthread(int i, Mat mGr, uchar *d, int row, int col, int *str_start_rgb,
                vector<vector<int> > str_sub_rgb, int rect_w, int rect_y) {
    LOGI("======mtestthread============================= ");

    LOGI("======mtestthread============================= 1");
    LOGI("======multipointFindColor==========================return============================= %d ;%d;%d",
         str_sub_rgb[0][0], str_sub_rgb[0][1], str_sub_rgb[0][2]);
}


void similar_arithmetic(Mat mGr, uchar *d, int row, int col, int *str_start_rgb,
                        vector<vector<int> > str_sub_rgb, vector<vector<int> > str_sub_xy,
                        int rect_w, int rect_y, int *range_, double sim, int *return_value) {
    double target = sim;
    int c1, c2, c3, c4, c12;
    int index;
    int threshold = (int) (765 * (1 - target));
    int up_sub[3] = {-1, -1, -1};
    int up_sub1[3] = {-1, -1, -1};
    vector<vector<int> > str_sub_xy_copy(str_sub_xy);
    vector<vector<int> > str_sub_rgb_copy(str_sub_rgb);
//    memcpy(str_sub_xy_copy, str_sub_xy, str_sub_xy.size()*2*4);

//    for (int i =0;i<str_sub_xy_copy.size();i++){
//        LOGI("======multipointFindColor=========================== %d ; %d ",str_sub_xy_copy[i][0],str_sub_xy[i][0]);
//    }



    for (int i = range_[1]; i < range_[3]; ++i) {
        d = mGr.ptr<uchar>(i);
        for (int j = range_[0]; j < range_[2]; ++j) {
            if (return_value[0] > -1) {
                return;
            }
            index = j * 3;
            int r1, g1, b1;
            r1 = d[index];
            g1 = d[index + 1];
            b1 = d[index + 2];
            if (up_sub[0] == r1 && up_sub[1] == g1 && up_sub[2] == b1) {
                continue;
            }
            if (r1 > str_start_rgb[0]) {
                c1 = r1 - str_start_rgb[0];
            } else {
                c1 = str_start_rgb[0] - r1;
            }
//            c1 = (c1 ^ (c1 >> 31)) - (c1 >> 31);
//            if (c1 > threshold) { continue; }
            if (g1 > str_start_rgb[1]) {
                c2 = g1 - str_start_rgb[1];
            } else {
                c2 = str_start_rgb[1] - g1;
            }
//            c2 = (c2 ^ (c2 >> 31)) - (c2 >> 31);

            c12 = c1 + c2;
//            if (c12 > threshold) { continue; }
            if (b1 > str_start_rgb[2]) {
                c3 = b1 - str_start_rgb[2];
            } else {
                c3 = str_start_rgb[2] - b1;
            }
//            c3 = (c3 ^ (c3 >> 31)) - (c3 >> 31);//绝对值
            c4 = c12 + c3;
            if (c4 < threshold) {
                up_sub[0] = -1;
                up_sub[1] = -1;
                up_sub[2] = -1;
                int x = j;
                int y = i;
                bool check = true;
                for (size_t k = 0; k < str_sub_xy.size(); ++k) {
                    int row = i + str_sub_xy[k][1];
                    int col = j + str_sub_xy[k][0];
                    if (row < 0 || row > rect_y || col < 0 || col > rect_w) {
                        check = false;
                        break;
                    }
                    index = col * 3;
                    r1 = mGr.ptr<uchar>(row)[index];
                    g1 = mGr.ptr<uchar>(row)[index + 1];
                    b1 = mGr.ptr<uchar>(row)[index + 2];
//                    if (up_sub1[0] == r1 && up_sub1[1] == g1 && up_sub1[2] == b1) {
//                        check = false;
//                        break;
//                    }
                    if (r1 > str_sub_rgb[k][0]) {
                        c1 = r1 - str_sub_rgb[k][0];
                    } else {
                        c1 = str_sub_rgb[k][0] - r1;
                    }
//                    if(c1 < 0){
//                        c1 = (c1 ^ (c1 >> 31)) - (c1 >> 31);
//                    }
                    if (c1 > threshold) {
                        check = false;
                        break;
                    }
                    c2 = g1 - str_sub_rgb[k][1];
                    c2 = (c2 ^ (c2 >> 31)) - (c2 >> 31);
                    c12 = c1 + c2;
                    if (c12 > threshold) {
                        check = false;
                        break;
                    }
                    c3 = b1 - str_sub_rgb[k][2];
                    c3 = (c3 ^ (c3 >> 31)) - (c3 >> 31);
                    c4 = c12 + c3;
                    if (c4 > threshold) {
//                        if (k > 0) {
//                            str_sub_xy[0][0] = str_sub_xy_copy[k][0];
//                            str_sub_xy[0][1] = str_sub_xy_copy[k][1];
//                            str_sub_xy[k][0] = str_sub_xy_copy[0][0];
//                            str_sub_xy[k][1] = str_sub_xy_copy[0][1];
//                            str_sub_xy_copy.assign(str_sub_xy.begin(), str_sub_xy.end());
//                            str_sub_rgb[0][0] = str_sub_rgb_copy[k][0];
//                            str_sub_rgb[0][1] = str_sub_rgb_copy[k][1];
//                            str_sub_rgb[0][2] = str_sub_rgb_copy[k][2];
//                            str_sub_rgb[k][0] = str_sub_rgb_copy[0][0];
//                            str_sub_rgb[k][1] = str_sub_rgb_copy[0][1];
//                            str_sub_rgb[k][2] = str_sub_rgb_copy[0][2];
//                            str_sub_rgb_copy.assign(str_sub_rgb.begin(), str_sub_rgb.end());
//                        }
//                        up_sub1[0] = r1;
//                        up_sub1[1] = g1;
//                        up_sub1[2] = b1;
                        check = false;
                        break;
                    }
//                    else {
//                        up_sub1[0] = -1;
//                        up_sub1[1] = -1;
//                        up_sub1[2] = -1;
//                    }

                }
                /*
                thread threads[str_sub_xy.size()];
                for (size_t k = 0; k < str_sub_xy.size(); k++) {
                    threads[k] = thread(sub_similar_arithmetic, i, j, str_sub_xy, str_sub_rgb,
                                        rect_y, rect_w, mGr, threshold, &check, k);
                }
                for (size_t k = 0; k < str_sub_xy.size(); k++) {
                    threads[k].join();
                }
                */

                if (check) {
//                    LOGI("======multipointFindColor==========================lock==============================");
                    my_lock.lock();
                    if (return_value[0] > -1) {
//                        LOGI("======multipointFindColor==========================return==============================");
                        my_lock.unlock();
                        return;
                    }
//                    LOGI("======multipointFindColor==========================return==============================%lf ;%d ;%d ",sim, x, y);
                    return_value[0] = x;
                    return_value[1] = y;
//                    LOGI("======multipointFindColor==========================unlock==============================");
                    my_lock.unlock();
                    return;
                }
            } else {
                up_sub[0] = r1;
                up_sub[1] = g1;
                up_sub[2] = b1;
            }
        }
    }
}


void similar_arithmetic_1(Mat mGr, uchar *d, int row, int col, int *str_start_rgb,
                          vector<vector<int> > str_sub_rgb, vector<vector<int> > str_sub_xy,
                          int rect_w, int rect_y, int *range_, double sim, int *return_value) {
    double target = sim;
    int c1, c2, c3, c4, c12;
    int index;
    int threshold = (int) (765 * (1 - target));
    int up_sub[3] = {-1, -1, -1};
    int up_sub1[3] = {-1, -1, -1};
    vector<vector<int> > str_sub_xy_copy(str_sub_xy);
    vector<vector<int> > str_sub_rgb_copy(str_sub_rgb);

    for (int i = range_[1]; i < range_[3]; ++i) {
        d = mGr.ptr<uchar>(i);
        for (int j = range_[0]; j < range_[2]; ++j) {
            if (return_value[0] > -1) {
                return;
            }
            index = j * 3;
            int r1, g1, b1;
            r1 = d[index];
            g1 = d[index + 1];
            b1 = d[index + 2];
            c1 = r1 - str_start_rgb[0];
            c1 = (c1 ^ (c1 >> 31)) - (c1 >> 31);
            c2 = g1 - str_start_rgb[1];
            c2 = (c2 ^ (c2 >> 31)) - (c2 >> 31);

            c12 = c1 + c2;
            c3 = b1 - str_start_rgb[2];
            c3 = (c3 ^ (c3 >> 31)) - (c3 >> 31);//绝对值
            c4 = c12 + c3;
            if (c4 < threshold) {
                int x = j;
                int y = i;
                bool check = true;
                for (size_t k = 0; k < str_sub_xy.size(); ++k) {
                    int row = i + str_sub_xy[k][1];
                    int col = j + str_sub_xy[k][0];
                    if (row < 0 || row >= rect_y || col < 0 || col >= rect_w) {
                        check = false;
                        break;
                    }
                    index = col * 3;
                    r1 = mGr.ptr<uchar>(row)[index];
                    g1 = mGr.ptr<uchar>(row)[index + 1];
                    b1 = mGr.ptr<uchar>(row)[index + 2];


                    c1 = r1 - str_sub_rgb[k][0];
                    c1 = (c1 ^ (c1 >> 31)) - (c1 >> 31);

                    c2 = g1 - str_sub_rgb[k][1];
                    c2 = (c2 ^ (c2 >> 31)) - (c2 >> 31);
                    c12 = c1 + c2;

                    c3 = b1 - str_sub_rgb[k][2];
                    c3 = (c3 ^ (c3 >> 31)) - (c3 >> 31);
                    c4 = c12 + c3;
                    if (c4 > threshold) {
                        if (k > 0) {
                            str_sub_xy[0][0] = str_sub_xy_copy[k][0];
                            str_sub_xy[0][1] = str_sub_xy_copy[k][1];
                            str_sub_xy[k][0] = str_sub_xy_copy[0][0];
                            str_sub_xy[k][1] = str_sub_xy_copy[0][1];
                            str_sub_xy_copy.assign(str_sub_xy.begin(), str_sub_xy.end());
                            str_sub_rgb[0][0] = str_sub_rgb_copy[k][0];
                            str_sub_rgb[0][1] = str_sub_rgb_copy[k][1];
                            str_sub_rgb[0][2] = str_sub_rgb_copy[k][2];
                            str_sub_rgb[k][0] = str_sub_rgb_copy[0][0];
                            str_sub_rgb[k][1] = str_sub_rgb_copy[0][1];
                            str_sub_rgb[k][2] = str_sub_rgb_copy[0][2];
                            str_sub_rgb_copy.assign(str_sub_rgb.begin(), str_sub_rgb.end());
                        }
                        check = false;
                        break;
                    }
                }
                /*
                thread threads[str_sub_xy.size()];
                for (size_t k = 0; k < str_sub_xy.size(); k++) {
                    threads[k] = thread(sub_similar_arithmetic, i, j, str_sub_xy, str_sub_rgb,
                                        rect_y, rect_w, mGr, threshold, &check, k);
                }
                for (size_t k = 0; k < str_sub_xy.size(); k++) {
                    threads[k].join();
                }
                */

                if (check) {
//                    LOGI("======multipointFindColor==========================lock==============================");
                    my_lock.lock();
                    if (return_value[0] > -1) {
//                        LOGI("======multipointFindColor==========================return==============================");
                        my_lock.unlock();
                        return;
                    }
//                    LOGI("======multipointFindColor==========================return==============================%lf ;%d ;%d ",sim, x, y);
                    return_value[0] = x;
                    return_value[1] = y;
//                    LOGI("======multipointFindColor==========================unlock==============================");
                    my_lock.unlock();
                    return;
                }
            }
        }
    }
}


void sub_similar_arithmetic(int i, int j, vector<vector<int> > str_sub_xy,
                            vector<vector<int> > str_sub_rgb, int rect_y, int rect_w, Mat mGr,
                            int threshold, bool *check, int k) {
    int c1, c2, c3, c4, index;
    int row = i + str_sub_xy[k][1];
    int col = j + str_sub_xy[k][0];
//LOGI("======multipointFindColor==========================1==============================%d %d ",row,col);
// LOGI("======multipointFindColor==========================1==============================%d %d ",row,col);
    if (row < 0 || row > rect_y || col < 0 || col > rect_w) {
        my_lock.lock();
        *check = false;
        my_lock.unlock();
        return;
    }
    index = col * 3;
    int r1 = mGr.ptr<uchar>(row)[index];
    int g1 = mGr.ptr<uchar>(row)[index + 1];
    int b1 = mGr.ptr<uchar>(row)[index + 2];
    c1 = r1 - str_sub_rgb[k][0];
    c1 = (c1 ^ (c1 >> 31)) - (c1 >> 31);
//                    if (c1>threshold){
//                        check = false;
//                        break;
//                    }
    c2 = g1 - str_sub_rgb[k][1];
    c2 = (c2 ^ (c2 >> 31)) - (c2 >> 31);
    c3 = b1 - str_sub_rgb[k][2];
    c3 = (c3 ^ (c3 >> 31)) - (c3 >> 31);
    c4 = c1 + c2 + c3;
//                    sim = c4 / 765.0;
    if (c4 > threshold) {
//        LOGI("======multipointFindColor==========================1==============================check  false ");
//                   if(sim < target){
        my_lock.lock();
        *check = false;
        my_lock.unlock();
//        LOGI("======multipointFindColor==========================unlock==============================check_sub ");
        return;
    }


}


JNIEXPORT void JNICALL
Java_com_padyun_lxh_jxqy_PublicFunction_ScreenCap(JNIEnv *env, jobject obj) {
    LOGI("======ScreenCap=======================================================");
//    sp<ISurfaceComposer> composer;
//    getService(name, &composer);
//    sp<IMemoryHeap> heap;
//    status_t err = composer->captureScreen(0, &heap, &w, &h, &f, 0, 0);




}

//#endif

JNIEXPORT void JNICALL
Java_com_padyun_publicfunctionlibrary_FunctionClass_test() {
    for (int i = 0; i < 720; ++i) {

        LOGI("======multipointFindColor===========y =  %d ",
             i);
        if (i == 500) {
            LOGI("======multipointFindColor=====508======y =  %d ",
                 i);
        }
    }
}
