python py_file/align_dataset_mtcnn.py  DataSet/inputImage DataSet/mtcnnImage --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25

python py_file/classifier.py TRAIN DataSet/mtcnnImage Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000
