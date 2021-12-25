# Face-Recognition-With-InsightFace

Để nhận diện khuôn mặt trong bức ảnh, mình sử dụng InsightFace. InsightFace có 2 bước đó là detect và recognize.<br>
Bức ảnh sau khi được detect sẽ cho ta các landmarks, rồi từ các landmarks cho ra một embedding (, 512).<br>
Sau khi có các embeddings và classes tương ứng, ta chỉ cần dùng một classifier là có thể nhận diện được ai trong bức ảnh.<br>

# Tổ chức dữ liệu

    \---file_name
        +---class1
        |   img1.jpg
        |   img2.jpg
        |   ...
        +---class2
        |   img1.jpg
        |   img2.jpg
        |   ...
        ...
 
 Với mỗi thư mục là một class, chứa dữ liệu, tên của thư mục là tên class.
 
# Install environments
    pip install -r requirements.txt
    
# Quick start

## Chuẩn bị dữ liệu
     python ./prepare_dataset.py

Lúc này, chương trình sẽ xuất ra 2 files X.npy và y.npy tương ứng với features và targets tại thư mục model.

## Train
    python ./train.py
    
Minh sử dụng thuật toán KNN với số lân cận là 3.<br>
Chương trình sẽ huấn luyện X và y, rồi xuất ra 1 pretrained model có tên mặc định là my_model.sav.

## Recognize
    python ./recognize_from_video.py --input-path ./testset/test.jpg

## Kết quả
![alt text](https://github.com/tdbui1209/Face-Recognition-With-InsightFace/blob/main/output/test.jpg)
Bức ảnh đầu ra mặc định là ở file output. Dữ liệu của ta không có Gina, Hitchcock, Scully, do đó cả 3 người họ đều hiển thị "Unknown".
