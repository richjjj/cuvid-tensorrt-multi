#ifndef AICALLBACK_H_
#define AICALLBACK_H_
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    alarm_connect    = 0,  // 连接成功
    alarm_disconnect = 1,  // 断开连接成功
    alarm_data       = 2,  // 数据信息
    alarm_control    = 3,  // 控制命令
} protocol_register_key_e;

// 回调函数
//////////////////////////////////////////////////////////////
// callbackType : 回调的数据类型，0:状态回调,1:数据回调
// img : cv::Mat
// datalen : 数据长度
// data : 数据内容
///////////////////////////////////////////////////////////////
typedef void (*CallBackDataInfo)(int callbackType, void *img, char *data, int datalen);
typedef void (*MessageCallBackDataInfo)(int callbackType, void *img, char *data, int datalen);

#ifdef __cplusplus
}
#endif

#endif