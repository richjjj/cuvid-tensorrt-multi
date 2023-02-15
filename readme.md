# docker 部署
- 镜像来源：pull ngc，编译opencv
- 容器启动命令 `docker run --name cangku_runtime -it --network host --gpus all -v /home/xmrbi/cangku_workspace:/cangku_workspace --workdir /cangku_workspace imagename "/opt/nvidia/nvidia_entrypoint.sh"`
- docker run --name intelligenttrafficsurveillance -it --network host --pids-limit -1 --gpus all -v/data:/data -v/root/intelligenttrafficsurveillance_workspace:/intelligenttrafficsurveillance_workspace --workdir /intelligenttrafficsurveillance_workspace xmrbi/cangku

# 使用
1. 为nvcuvid创建软链接，这个库随显卡驱动发布
    - `bash link-cuvid.sh`
1. 若为jetson平台，指定 Makefile `make -f Makefile_jetson jetson`,移除不支持的解码库
2. 编译libpro.so, `bear make all` 生成 compile_commands.json
    - `make all -j64`
3. 执行对比测试`make yolo -j64`
4. 执行解封装测试`make demuxer -j64`
5. 执行硬件解码测试`make hard_decode -j64`
6. 执行pipeline `make pipeline -j64`

# 如果要在目录下执行
- 请执行
    ```bash
    source ``
    ```

# TODO
- [ ] 定义require，从http服务下载onnx文件和第三方依赖库
