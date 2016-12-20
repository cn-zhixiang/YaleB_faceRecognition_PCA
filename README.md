# YaleB_faceRecognition_PCA<br/>
基于特征脸对 YaleB 数据集的人脸识别，IDE 为 Octave。项目采用了「矢量化编程」，循环相对较少。<br/>
<br/>
SRBFR.m 是项目的主体，输入数据集地址和每个人需要训练的张数，输出准确率。<br/>
PCA.m 是对 Matlab 中 princomp 函数的重写。<br/>
feature_sign.m 返回 test 样本对之前得到的特征向量的系数向量，该系数向量可使残差（噪声）最小<br/>
