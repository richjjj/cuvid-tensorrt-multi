# docker 部署
- 镜像来源：ngc, `docker pull nvcr.io/nvidia/tensorrt:22.08-py3`，安装opencv
- docker run

# 使用
1. 为nvcuvid创建软链接，这个库随显卡驱动发布
    - `bash link-cuvid.sh`
1. 若为jetson平台，指定 Makefile `make -f Makefile_jetson jetson`,移除不支持的解码库
2. 编译libpro.so。（`bear make all` 生成 compile_commands.json）
    - `make all -j64`
3. 执行对比测试`make yolo -j64`
4. 执行解封装测试`make demuxer -j64`
5. 执行硬件解码测试`make hard_decode -j64`
6. 查看 makefile 执行不同的例子 `make test`

# TODO
- [ ] 定义require，从http服务下载onnx文件和第三方依赖库
