# OFDM MIMO Channel Estimation and Detection with Sionna (5G)
## 1. Mục tiêu
Dự án này tập trung nghiên cứu và mô phỏng bài toán ước lượng kênh (Channel Estimation) và phát hiện tín hiệu (MIMO Detection) trong hệ thống OFDM-MIMO theo chuẩn 5G NR, sử dụng framework Sionna của NVIDIA dựa trên TensorFlow.
Các mục tiêu chính của dự án bao gồm:

Xây dựng mô hình end-to-end OFDM-MIMO từ phía phát đến phía thu theo chuẩn 5G

Nghiên cứu và so sánh các phương pháp ước lượng kênh OFDM, bao gồm:

- Least Squares (LS)

- Nội suy Nearest-Neighbor

- Nội suy Linear

- Nội suy LMMSE (có/không làm mượt không gian – thời gian – tần số)

Đánh giá độ chính xác ước lượng kênh thông qua chỉ số MSE (Mean Square Error) theo SNR

So sánh các thuật toán MIMO Detection phổ biến trong Sionna:

- LMMSE (Linear Detection)

- K-Best Detection

- Expectation Propagation (EP)

- MMSE-PIC

Phân tích ảnh hưởng của:

- Perfect CSI vs Imperfect CSI (Channel Estimation)

- SNR (Eb/N0)

- Số anten MIMO 

Đánh giá hiệu năng hệ thống thông qua:

- SER (Symbol Error Rate)

- BER (Bit Error Rate) (có mã hóa LDPC 5G)

Dự án hướng tới mục tiêu benchmark các bộ thu OFDM-MIMO trong điều kiện thực tế, đồng thời làm nền tảng cho các nghiên cứu nâng cao như learned receivers hoặc model-driven deep learning trong 5G/6G.

## 2. Cài đặt
Phần này hướng dẫn chi tiết cách cài đặt môi trường để chạy mô phỏng OFDM MIMO Channel Estimation and Detection bằng framework Sionna, đảm bảo tái lập được toàn bộ kết quả mô phỏng trong dự án.

### 2.1. Yêu cầu hệ thống

2.1.1. Phần cứng (khuyến nghị)

CPU: Intel/AMD 64-bit

RAM: ≥ 8 GB (khuyến nghị ≥ 16 GB)

GPU: NVIDIA GPU hỗ trợ CUDA (tùy chọn, nhưng giúp tăng tốc mô phỏng)

Dung lượng trống: ≥ 10 GB

2.1.2. Phần mềm

Hệ điều hành:
- Ubuntu 20.04 / 22.04 / 24.04 (khuyến nghị)

Python:

- Phiên bản Python 3.9 – 3.10

Trình soạn thảo / môi trường phát triển:

- VS Code, PyCharm hoặc Jupyter Notebook

2.2. Cài đặt Python và môi trường ảo

Khuyến nghị sử dụng virtual environment để tránh xung đột thư viện.

2.2.1. Cài đặt Python (nếu chưa có)

```
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```
Kiểm tra phiên bản:
```
python3 --version
```
2.2.2. Tạo và kích hoạt môi trường ảo

```
python3 -m venv venv
source venv/bin/activate
```
Sau khi kích hoạt thành công, terminal sẽ hiển thị tiền tố (venv).

2.3. Cài đặt các thư viện cần thiết
2.3.1. Cập nhật pip
```
pip install --upgrade pip
```
2.3.2. Cài đặt Sionna và các thư viện phụ thuộc
```
pip install sionna tensorflow numpy scipy matplotlib
```

Các phiên bản thư viện đã được sử dụng trong dự án trên Ubuntu 22.04:
---
| Thư viện       | Phiên bản |
| -------------- | --------- |
| **Sionna**     | 1.2.0     |
| **TensorFlow** | 2.10.1    |
| **NumPy**      | 1.26.4    |
| **SciPy**      | 1.11.4    |
| **Matplotlib** | 3.8.2     |

---

2.4. Cài đặt và sử dụng Jupyter Notebook (nếu dùng)
```
pip install jupyterlab
```

Truy cập Jupyter Notebook trong môi trường ảo:
```
jupyter lab
``
