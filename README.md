# yolov7-opencv-dnn-cpp
使用opencv模块部署yolov7-0.1版本和yolov5-6.0以上版本<br>

+ 基于yolov5-6.0版本的yolov5:https://github.com/ultralytics/yolov5 <br>
+ 基于yolov7-0.1的版本https://github.com/WongKinYiu/yolov7 <br>

OpenCV>=4.5.0 <br>

> python path/to/export.py --weights yolov5s.pt --img [640,640] --opset 12 --include onnx<br>
> python path/to/export.py --weights yolov7.pt --img [640,640] <br>
请注意，yolov7导出的时候不要加--grid这个参数(控制detect层),否则opencv读取没问题，推理报错.<br>

可以通过yolo.h中定义的YOLOV5宏定义来切换yolov5和yolov7两个版本，(其实两个版本onnx后处理方式差不多的说<br>
>通过yolo.h中定义的YOLO_P6来切换是否使用两者的P6模型。<br>
> YOLOV5:true -->yolov5.onnx<br>
> YOLOV5:false-->yolov7.onnx

2022-09-06 updata:<br>
最近有些小伙伴使用opencv4.6的版本报错了，经过debug发现，opencv4.6的和4.5.x的forward输出顺序不一样导致的，使用opencv4.6的时候在net.forward之后需要加上一个排序，使得输出口从大到小排序才行。<br>
https://github.com/UNeedCryDear/yolov7-opencv-dnn-cpp/blob/79ac7a93d4ec00bc06295a481b1dcc22893f97e2/yolo.cpp#L48

另外关于换行符，windows下面需要设置为CRLF，上传到github会自动切换成LF，windows下面切换一下即可。<br>
贴个yolov7.onnx和yolov5s.onnx的对比<br>
![yolo](https://user-images.githubusercontent.com/52729998/180824922-0c7dc3f9-fbda-497b-9ae3-3f299b8c0452.png)
