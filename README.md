# HNSW Semantic Document Search Engine
### Hệ Thống Tìm Kiếm Tài Liệu Theo Ngữ Nghĩa Bằng Thuật Toán HNSW (Python + Không Gian 3D Thời Gian Thực + Tách Chữ PDF)

Hệ thống được thiết kế và xây dựng chuẩn mực theo lý thuyết đồ thị **Hierarchical Navigable Small World (HNSW)** của tác giả **Yu. A. Malkov & D. A. Yashunin (IEEE TPAMI 2018)** kết hợp với mô hình nhúng ngữ nghĩa sâu **all-MiniLM-L6-v2 (384 chiều)** tăng tốc trên card đồ họa **NVIDIA CUDA GPU**, tích hợp **Không gian hiển thị 3D WebGL (Three.js)** với dữ liệu thật và thời gian thực.

---

## 🌟 Tính Năng Nổi Bật

1. **Không Gian Hiển Thị 3D Thời Gian Thực (Three.js WebGL & OrbitControls)**:
   - **Tọa độ ngữ nghĩa thực tế**: Toàn bộ vector 384 chiều được chiếu xuống mặt phẳng 2D $(X, Z)$ bằng thuật toán **PCA (Principal Component Analysis)** từ `scikit-learn`. Các tài liệu có ngữ nghĩa tương đồng (vd: AI, Deep Learning) sẽ tự động cụm lại gần nhau trong không gian vật lý 3D.
   - **Trục tung $Y$ đại diện cho các tầng HNSW**: Các mặt phẳng $L_0, L_1, L_2...$ trôi nổi lơ lửng với lưới tọa độ bán trong suốt và màu sắc nhận diện đặc trưng (Tầng đáy Emerald, Tầng trung gian Blue, Tầng đỉnh Coral/Red).
   - **Cột gióng đứng liên tầng (Vertical Identity Pillars)**: Các đường nét đứt 3D kết nối các bản thể của cùng một nút dữ liệu từ tầng cao nhất xuống tầng đáy $L_0$.
   - **Mô phỏng đường chạy Routing thời gian thực**: Khi nhập một câu hỏi, điểm truy vấn $\mathbf{Q}$ xuất hiện dưới dạng một khối đa diện hoàng kim phát sáng. Tia sáng quét 3D sẽ nhảy qua các nút theo thuật toán greedy routing ở tầng trên, hạ độ cao qua cột gióng đứng, và quét chùm tia (beam search) tại tầng $L_0$.
   - **Bảng điều khiển tương tác**: Cho phép Phát/Tạm dừng/Tua từng bước nhảy (`Step-by-step Scrubbing`), xoay 360°, thu phóng, chuyển đổi góc nhìn Camera (Toàn cảnh Isometric, Từ trên xuống Top-Down, Tập trung L0).
   - **Raycasting Node Inspector**: Nhấp chuột vào bất kỳ khối cầu nào trong không gian 3D để xem tiêu đề tài liệu, danh mục, tọa độ và tóm tắt nội dung.

2. **Cài Đặt Thuật Toán HNSW Thuần Python Từ Gốc (`hnsw.py`)**:
   - Phân tầng ngẫu nhiên theo phân phối hình học: $l = \lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \rfloor$ với $m_L = 1 / \ln(M)$.
   - Tìm kiếm chùm tia đa tầng (Multi-layer Priority Queue Beam Search) sử dụng min-heap và max-heap.
   - **Heuristic chọn lân cận Đa hướng (RNG / Empty-Lune condition)**: Triệt tiêu các liên kết dư thừa tạo tam giác tù, bảo toàn tính định hướng đa chiều trong không gian $d=384$.
   - Liên kết hai chiều (Bidirectional Linking) và cắt tỉa bậc tối đa ($M, M_0$).
   - Thu thập toàn bộ vết định tuyến (`trace`): ghi lại từng bước nhảy, các nút được duyệt và trạng thái chấp nhận vào chùm tia.

3. **Chức Năng Tách Chữ Từ PDF & Nạp CSDL (`pdf_processor.py`)**:
   - Sử dụng thư viện **PyMuPDF (`fitz`)** tốc độ cao, hỗ trợ đầy đủ font chữ Unicode tiếng Việt.
   - Hỗ trợ kéo thả file PDF bất kỳ từ máy tính.
   - Tùy chọn tách theo từng trang (page-by-page) hoặc ghép toàn bộ văn bản.
   - Tự động sinh vector trên PyTorch GPU và chèn động vào đồ thị HNSW ngay lập tức.
   - Nút 1-click **"Thử Nghiệm Tải PDF Báo Cáo HNSW (45 Trang Có Sẵn)"** để nạp ngay tài liệu nghiên cứu chuyên sâu có sẵn trong dự án.

4. **Giao Diện Sáng Mặc Định (Light Theme by Default)**:
   - Thiết kế hiện đại, tinh tế với nền sáng dịu mắt (`slate-50`), thẻ trắng nổi bật (`bg-white`) và nút chuyển đổi Sáng/Tối (☀️/🌙).

---

## 🚀 Hướng Dẫn Khởi Chạy

```powershell
python run_semantic_search.py
```
Hệ thống sẽ tự động khởi động máy chủ Flask tại cổng `5000` và tự động mở trình duyệt web tại địa chỉ:
👉 **`http://localhost:5000`** (hoặc `http://127.0.0.1:5000`)

---

## 📁 Cấu Trúc Mã Nguồn

```
HNSW/
├── run_semantic_search.py       # Script khởi chạy 1-click
└── semantic_search/
    ├── hnsw.py                  # Thuật toán HNSW đầy đủ từ nguyên lý toán học
    ├── embedder.py              # Bộ tạo vector ngữ nghĩa PyTorch CUDA
    ├── corpus.py                # Kho dữ liệu văn bản và quản lý tài liệu
    ├── pdf_processor.py         # Bộ tách chữ PDF và phân đoạn tài liệu (PyMuPDF)
    ├── app.py                   # Máy chủ Flask REST API & chiếu tọa độ 3D PCA
    ├── cli.py                   # Công cụ tìm kiếm CLI dòng lệnh
    ├── static/
    │   ├── three.min.js         # Thư viện đồ họa WebGL Three.js offline
    │   └── OrbitControls.js     # Bộ điều khiển camera xoay 360°
    └── templates/
        └── index.html           # Giao diện Web SPA tích hợp Viewport 3D Three.js
```
