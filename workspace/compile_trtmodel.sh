trtexec --onnx=$1.onnx --saveEngine=$1.trtmodel --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:32x3x640x640
