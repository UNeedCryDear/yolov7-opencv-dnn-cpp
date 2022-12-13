# yolov7-opencv-dnn-cpp
使用opencv模块部署yolov7-0.1版本和yolov5-6.0以上版本<br>

+ 基于yolov5-6.0版本的yolov5:https://github.com/ultralytics/yolov5 <br>
+ 基于yolov7-0.1的版本https://github.com/WongKinYiu/yolov7 <br>

## **OpenCV>=4.5.0** <br>

> python path/to/export.py --weights yolov5s.pt --img [640,640] --opset 12 --include onnx<br>
> python path/to/export.py --weights yolov7.pt --img [640,640] <br>
请注意，yolov7导出的时候不要加--grid这个参数(控制detect层),否则opencv读取没问题，推理报错.<br>

可以通过yolo.h中定义的YOLOV5宏定义来切换yolov5和yolov7两个版本，(其实两个版本onnx后处理方式差不多的说<br>
>通过yolo.h中定义的YOLO_P6来切换是否使用两者的P6模型。<br>
> YOLOV5:true -->yolov5.onnx<br>
> YOLOV5:false-->yolov7.onnx
#### 2022.12.13 更新：
+ 如果你的显卡支持FP16推理的话，可以将模型读取代码中的```DNN_TARGET_CUDA```改成```DNN_TARGET_CUDA_FP16```提升推理速度（虽然是蚊子腿，好歹也是肉（： 
#### 2022-09-06 更新:
+ 最近有些小伙伴使用opencv4.6的版本报错了，经过debug发现，opencv4.6的和4.5.x的forward输出顺序不一样导致的，使用opencv4.6的时候在net.forward之后需要加上一个排序，使得输出口从大到小排序才行。<br>
https://github.com/UNeedCryDear/yolov7-opencv-dnn-cpp/blob/86d4f5ef6ecfd7eb36a14d0c06a84a5468ff98e6/yolo.cpp#L48
#### 2022-10-18 更新:
yolov7目前有些模型低于opencv4.5.5会报错,报错信息类似下面使用opencv4.5.0读取yolov7-d6.pt转出的onnx模型（不能加参数--grid），此时建议升级下opencv的版本
> >OpenCV(4.5.0) Error: Unspecified error (> Node [Slice]:(341) parse error: OpenCV(4.5.0) D:\opencv\ocv4.5.0\sources\modules\dnn\src\onnx\onnx_importer.cpp:697: error: (-2:Unspecified error) in function 'void __cdecl cv::dnn::dnn4_v20200908::ONNXImporter::handleNode(const class opencv_onnx::NodeProto &)'</br>
> > Slice layer only supports steps = 1 (expected: 'countNonZero(step_blob != 1) == 0'), where</br>
> >     'countNonZero(step_blob != 1)' is 1</br>
> > must be equal to</br>
> >     '0' is 0</br>
> > in cv::dnn::dnn4_v20200908::ONNXImporter::handleNode, file D:\opencv\ocv4.5.0\sources\modules\dnn\src\onnx\onnx_importer.cpp, line 1797</br>

debug可以发现是由于yolov7-d6中使用了ReOrg模块引起的报错，这个模块有点类似早期的yolov5的Facos模块，如果一定要在opencv4.5.0下面运行，需要将ReOrg模块修改成下面的代码。
在models/common.py里面搜索下ReOrg.
```
class ReOrg(nn.Module):
    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        #origin code
        # return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        self.concat=Contract(gain=2)
        return self.concat(x)
```



另外关于换行符，windows下面需要设置为CRLF，上传到github会自动切换成LF，windows下面切换一下即可。<br>
贴个yolov7.onnx和yolov5s.onnx的对比<br>
![yolo](https://user-images.githubusercontent.com/52729998/180824922-0c7dc3f9-fbda-497b-9ae3-3f299b8c0452.png)
