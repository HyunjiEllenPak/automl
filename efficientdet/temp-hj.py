import tensorflow as tf
i=0
for f in  tf.io.gfile.glob('D:/imageData/cocodata/test2017/test2017/*.jpg'):
    i+=1
    print(f)
print(i)