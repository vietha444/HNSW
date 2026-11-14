"""
Semantic Document Corpus & Management
=====================================
Provides a rich, diverse bilingual knowledge base across 7 major domains:
1. Artificial Intelligence & Deep Learning
2. Vector Search & HNSW Graph Indexing
3. Distributed Systems & Cloud Engineering
4. Quantum Computing & Physics
5. Biomedicine & Genomics
6. Culture, History & Gastronomy
7. FinTech & Economics
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple

DEFAULT_DOCUMENTS = [
    # --- DOMAIN 1: AI & Deep Learning ---
    {
        "id": 0,
        "title": "Kiến Trúc Mô Hình Biến Đổi Attention và Transformer",
        "category": "AI & Deep Learning",
        "tags": ["Transformer", "Attention", "NLP", "LLM", "Deep Learning"],
        "summary": "Phân tích cơ chế Self-Attention trong Transformer, loại bỏ hoàn toàn tính tuần tự của RNN.",
        "content": """Kiến trúc Transformer được giới thiệu trong bài báo kinh điển 'Attention Is All You Need' năm 2017 bởi nhóm tác giả Google Research. Cốt lõi của mô hình là cơ chế Self-Attention đa đầu (Multi-Head Attention), cho phép mô hình tính toán trọng số tương tác giữa mọi cặp từ trong câu cùng một lúc. Khác với kiến trúc Mạng nơ-ron hồi quy (RNN) hay LSTM vốn xử lý tuần tự từng bước thời gian, Transformer có khả năng song song hóa cực kỳ hiệu quả trên phần cứng GPU, mở đường cho sự bùng nổ của các mô hình ngôn ngữ lớn (LLM) hiện đại như BERT, GPT-4, LLaMA và Gemini."""
    },
    {
        "id": 1,
        "title": "Thuật Toán Lan Truyền Ngược và Gradient Descent",
        "category": "AI & Deep Learning",
        "tags": ["Backpropagation", "Gradient Descent", "Optimization", "Neural Networks"],
        "summary": "Nguyên lý huấn luyện mạng nơ-ron nhân tạo thông qua tính toán đạo hàm chuỗi.",
        "content": """Thuật toán lan truyền ngược (Backpropagation) là nền tảng cốt lõi của việc huấn luyện mạng nơ-ron sâu. Bằng cách áp dụng quy tắc đạo hàm chuỗi (chain rule) của giải tích, thuật toán tính toán đạo hàm riêng của hàm mất mát (loss function) đối với từng trọng số trong toàn bộ các tầng mạng. Sau đó, các giải thuật tối ưu hóa dựa trên Gradient Descent như SGD, Adam, AdamW sẽ cập nhật trọng số theo hướng ngược chiều gradient để cực tiểu hóa sai số dự đoán."""
    },
    {
        "id": 2,
        "title": "Mạng Nơ-ron Tích Chập (CNN) Trong Thị Giác Máy Tính",
        "category": "AI & Deep Learning",
        "tags": ["CNN", "Computer Vision", "Convolution", "Image Recognition"],
        "summary": "Cơ chế trích xuất đặc trưng không gian đa tỷ lệ trong ảnh bằng phép tích chập.",
        "content": """Mạng nơ-ron tích chập (Convolutional Neural Networks - CNN) tận dụng cấu trúc không gian 2 chiều của hình ảnh thông qua các bộ lọc hạt nhân (kernel filters) có thể học được. Các tầng tích chập ban đầu nhận diện các đặc trưng cấp thấp như đường biên, góc cạnh và vân bề mặt, trong khi các tầng sâu hơn tổng hợp thành các bộ phận vật thể và khái niệm ngữ nghĩa phức tạp. Kiến trúc ResNet với các kết nối tắt (residual connections) đã giải quyết triệt để vấn đề biến mất gradient khi mạng trở nên cực sâu."""
    },
    {
        "id": 3,
        "title": "Học Tăng Cường Từ Phản Hồi Con Người (RLHF)",
        "category": "AI & Deep Learning",
        "tags": ["RLHF", "Reinforcement Learning", "Alignment", "LLM"],
        "summary": "Phương pháp tinh chỉnh các mô hình ngôn ngữ lớn để căn chỉnh theo ý định và đạo đức con người.",
        "content": """Học tăng cường từ phản hồi con người (Reinforcement Learning from Human Feedback - RLHF) là kỹ thuật quan trọng giúp đưa các mô hình ngôn ngữ nền tảng thành các trợ lý đàm thoại an toàn và hữu ích. Quy trình gồm 3 bước: huấn luyện mô hình hành vi có giám sát (SFT), xây dựng mô hình phần thưởng (Reward Model) học từ các cặp câu trả lời được con người xếp hạng, và tối ưu hóa chính sách sinh từ bằng thuật toán Proximal Policy Optimization (PPO) để tối đa hóa phần thưởng mà không làm suy thoái năng lực ngôn ngữ."""
    },

    # --- DOMAIN 2: Vector Search & HNSW ---
    {
        "id": 4,
        "title": "Giải Mã Thuật Toán HNSW: Cấu Trúc Đồ Thị Nhỏ Điều Hướng Phân Tầng",
        "category": "Vector Search & Indexing",
        "tags": ["HNSW", "ANN", "Vector Search", "Small World", "Graph"],
        "summary": "Nguyên lý tìm kiếm lân cận gần đúng $O(\\log N)$ qua cấu trúc đồ thị đa tầng lồng nhau.",
        "content": """Hierarchical Navigable Small World (HNSW) do Yu. A. Malkov và D. A. Yashunin sáng tạo là thuật toán tìm kiếm lân cận gần đúng (ANN) dựa trên đồ thị hàng đầu thế giới. Thuật toán lấy cảm hứng từ cấu trúc phân tầng của mạng lưới Skip-List nhưng mở rộng sang không gian đồ thị đa chiều. Các tầng trên cùng thưa thớt đóng vai trò cao tốc định tuyến với khoảng cách nhảy dài, trong khi tầng đáy (Layer 0) dày đặc chứa toàn bộ dữ liệu để tiến hành tìm kiếm chùm tia (beam search) cục bộ, đạt tốc độ truy vấn logarit O(log N) với Recall tiệm cận 100%."""
    },
    {
        "id": 5,
        "title": "Điều Kiện Empty-Lune và Heuristic Chọn Lân Cận Đa Hướng",
        "category": "Vector Search & Indexing",
        "tags": ["HNSW", "RNG", "Delaunay", "Heuristic", "Graph Topology"],
        "summary": "Thuật toán thu hẹp lân cận để bảo toàn tính đa dạng góc và tránh tam giác tù trong HNSW.",
        "content": """Một trong những đóng góp mang tính đột phá của HNSW là Heuristic chọn lân cận dựa trên Đồ thị Lân cận Tương đối (Relative Neighborhood Graph - RNG). Thay vì chỉ kết nối thuần túy với các điểm gần nhất (vốn gây co cụm một hướng và lãng phí liên kết), HNSW áp dụng điều kiện 'vùng lưỡi liềm trống' (empty-lune condition): một điểm ứng viên chỉ được chấp nhận nếu khoảng cách từ nó tới điểm truy vấn nhỏ hơn khoảng cách từ nó tới bất kỳ lân cận nào đã được chọn trước đó. Nhờ vậy đồ thị luôn duy trì các cạnh tỏa ra mọi hướng không gian."""
    },
    {
        "id": 6,
        "title": "Kiến Trúc Cơ Sở Dữ Liệu Vector Phân Tán Milvus và Qdrant",
        "category": "Vector Search & Indexing",
        "tags": ["Vector Database", "Milvus", "Qdrant", "RAG", "Distributed"],
        "summary": "Thiết kế hệ thống lưu trữ và truy vấn hàng tỷ vector embedding phục vụ AI thế hệ mới.",
        "content": """Cơ sở dữ liệu vector hiện đại như Milvus, Qdrant và Vespa được xây dựng để lưu trữ, lập chỉ mục và truy vấn hàng tỷ vector embedding thời gian thực. Hệ thống phân tách rõ rệt giữa tầng điều phối truy vấn (Query Nodes), tầng xây dựng chỉ mục (Index Nodes) và tầng lưu trữ dữ liệu (Segment Storage). Bên cạnh các chỉ mục đồ thị như HNSW, các hệ thống này kết hợp lọc siêu dữ liệu (Scalar Filtering) và nén vector (Product Quantization - PQ) để tối ưu dung lượng RAM và băng thông mạng."""
    },
    {
        "id": 7,
        "title": "Tối Ưu Hóa FAISS Trên GPU Với Lõi Tính Toán CUDA",
        "category": "Vector Search & Indexing",
        "tags": ["FAISS", "CUDA", "GPU", "SIMT", "Parallelism"],
        "summary": "Giải quyết nghịch lý phân kỳ luồng (Warp Divergence) khi chạy duyệt đồ thị trên kiến trúc SIMT.",
        "content": """Thư viện Facebook AI Similarity Search (FAISS) cung cấp các thuật toán tìm kiếm vector tăng tốc phần cứng mạnh mẽ nhất hiện nay. Mặc dù các thuật toán duyệt đồ thị như HNSW thường gặp khó khăn trên GPU do tính rẽ nhánh ngẫu nhiên không liên tục (gây Warp Divergence), FAISS giải quyết bằng kỹ thuật Warp-Collaborative Parallelism: thay vì gán mỗi luồng xử lý một bước nhảy, toàn bộ warp (32 threads) cùng hợp tác tính toán khoảng cách vector và đồng bộ danh sách ưu tiên trong bộ nhớ dùng chung (Shared Memory), khai thác tối đa băng thông bộ nhớ HBM3."""
    },

    # --- DOMAIN 3: Distributed Systems ---
    {
        "id": 8,
        "title": "Định Lý CAP và Mô Hình Tính Nhất Quán Trong Hệ Phân Tán",
        "category": "Distributed Systems",
        "tags": ["CAP Theorem", "Consistency", "Availability", "Distributed Systems"],
        "summary": "Sự đánh đổi không thể tránh khỏi giữa Tính nhất quán, Tính sẵn sàng và Khả năng chịu lỗi phân vùng.",
        "content": """Định lý CAP của Eric Brewer chỉ ra rằng một hệ thống phân tán chia sẻ dữ liệu chỉ có thể thỏa mãn tối đa hai trong ba thuộc tính cùng một thời điểm: Tính nhất quán (Consistency - mọi nút nhìn thấy cùng một dữ liệu tại một thời điểm), Tính sẵn sàng (Availability - mọi yêu cầu không lỗi đều nhận được phản hồi), và Khả năng chịu lỗi phân vùng (Partition Tolerance - hệ thống vẫn hoạt động khi mạng bị ngắt kết nối giữa các nút). Do lỗi mạng trên thực tế là không thể tránh khỏi (P luôn phải giữ), các hệ thống bắt buộc phải chọn giữa CP (như HBase, Zookeeper) hoặc AP (như Cassandra, DynamoDB)."""
    },
    {
        "id": 9,
        "title": "Thuật Toán Đồng Thuận Raft: Đơn Giản Hóa Quản Lý Log Nhân Bản",
        "category": "Distributed Systems",
        "tags": ["Raft", "Consensus", "Paxos", "Distributed Systems", "Fault Tolerance"],
        "summary": "Cơ chế bầu chọn lãnh đạo và nhân bản bản ghi trạng thái trong các cụm máy chủ.",
        "content": """Thuật toán đồng thuận Raft được Ongaro và Ousterhout phát minh tại Đại học Stanford nhằm giải quyết sự phức tạp khó hiểu của Paxos. Raft phân tách bài toán đồng thuận thành ba bài toán con độc lập: Bầu chọn lãnh đạo (Leader Election) bằng cơ chế Randomized Timer, Nhân bản log (Log Replication) có giám sát của lãnh đạo, và An toàn trạng thái (Safety). Nhờ tính trực quan và dễ cài đặt, Raft nhanh chóng trở thành tiêu chuẩn vàng trong các hệ thống phân tán cốt lõi như Kubernetes etcd, HashiCorp Consul và TiKV."""
    },
    {
        "id": 10,
        "title": "Kiến Trúc Hướng Sự Kiện và Hàng Đợi Tin Nhắn Apache Kafka",
        "category": "Distributed Systems",
        "tags": ["Kafka", "Event-Driven", "Streaming", "Distributed Log"],
        "summary": "Hệ thống nhật ký cam kết phân tán xử lý hàng triệu sự kiện mỗi giây với độ trễ mili-giây.",
        "content": """Apache Kafka là nền tảng streaming phân tán dựa trên mô hình nhật ký nối tiếp phân vùng (Partitioned Commit Log). Bằng cách tận dụng kỹ thuật Zero-Copy (gửi trực tiếp dữ liệu từ page cache của hệ điều hành vào Network Socket qua lệnh sendfile) cùng với việc ghi tuần tự trên đĩa cứng (Sequential I/O), Kafka có thể đạt thông lượng hàng trăm megabytes mỗi giây trên mỗi máy chủ. Kafka đóng vai trò là xương sống kết nối hàng trăm vi dịch vụ trong các kiến trúc hướng sự kiện (Event-Driven Architecture) quy mô lớn."""
    },

    # --- DOMAIN 4: Quantum Computing & Physics ---
    {
        "id": 11,
        "title": "Nguyên Lý Chồng Chập Lượng Tử và Khái Niệm Qubit",
        "category": "Quantum & Physics",
        "tags": ["Quantum Computing", "Qubit", "Superposition", "Physics"],
        "summary": "Cách các bit lượng tử biểu diễn đồng thời trạng thái 0 và 1 trên mặt cầu Bloch.",
        "content": """Trong khi một bit cổ điển chỉ có thể nhận giá trị nhị phân rời rạc 0 hoặc 1, một Qubit lượng tử có thể tồn tại ở trạng thái chồng chập tuyến tính (Linear Superposition) của cả hai trạng thái cơ sở, được mô tả bởi hàm sóng trên không gian Hilbert và trực quan hóa qua quả cầu Bloch. Nhờ tính chất chồng chập kết hợp với hiện tượng vướng víu lượng tử (Quantum Entanglement), một máy tính lượng tử gồm N qubit có thể đồng thời xử lý một không gian trạng thái có kích thước 2 mũ N chiều, mang lại ưu thế vượt trội trong các bài toán mô phỏng phân tử và tối ưu hóa tổ hợp."""
    },
    {
        "id": 12,
        "title": "Thuật Toán Lượng Tử Shor: Mối Đe Dọa Đối Với Mật Mã Khóa Công Khai RSA",
        "category": "Quantum & Physics",
        "tags": ["Shor Algorithm", "Cryptography", "RSA", "Quantum Computing"],
        "summary": "Khả năng phân tích thừa số nguyên lớn trong thời gian đa thức bằng phép biến đổi Fourier lượng tử.",
        "content": """Thuật toán Peter Shor công bố năm 1994 đã chứng minh rằng một máy tính lượng tử có khả năng chịu lỗi (Fault-Tolerant Quantum Computer) có thể phân tích một số nguyên lớn thành các thừa số nguyên tố trong thời gian đa thức O((log N)^3), thông qua việc tìm chu kỳ của hàm số bằng Phép biến đổi Fourier lượng tử (QFT). Điều này đe dọa trực tiếp đến tính an toàn của các hệ thống mật mã khóa công khai phổ biến nhất thế giới hiện nay như RSA và Đường cong Elliptic (ECC), thúc đẩy sự ra đời của mật mã hậu lượng tử (Post-Quantum Cryptography)."""
    },

    # --- DOMAIN 5: Biomedicine & Genomics ---
    {
        "id": 13,
        "title": "AlphaFold 3 và Cách Mạng Dự Đoán Cấu Trúc Không Gian Protein",
        "category": "Biomedicine & Genomics",
        "tags": ["AlphaFold", "Protein Folding", "Bioinformatics", "DeepMind", "AI for Science"],
        "summary": "Ứng dụng học sâu dự đoán cấu trúc 3D của các đại phân tử sinh học từ chuỗi axit amin.",
        "content": """Sự ra đời của hệ thống AlphaFold bởi Google DeepMind đã giải quyết triệt để bài toán sinh học phân tử thách thức nhân loại suốt 50 năm: xác định cấu trúc gấp cuộn 3 chiều của protein chỉ từ trình tự chuỗi axit amin 1D. AlphaFold tích hợp các kiến trúc Evoformer và mô hình khuếch tán không gian (Diffusion Module) để đồng thời mô hình hóa sự tiến hóa đồng biến và hình học không gian Euclide. Phiên bản AlphaFold 3 mở rộng khả năng dự đoán cấu trúc tương tác của protein với DNA, RNA và các phân tử thuốc tiềm năng."""
    },
    {
        "id": 14,
        "title": "Công Nghệ Chỉnh Sửa Gen CRISPR-Cas9 và Y Học Chính Xác",
        "category": "Biomedicine & Genomics",
        "tags": ["CRISPR", "Gene Editing", "Genomics", "Biotechnology"],
        "summary": "Chiếc kéo phân tử cho phép chỉnh sửa chính xác các đoạn mã DNA trong tế bào sống.",
        "content": """Hệ thống CRISPR-Cas9, khám phá đoạt giải Nobel Hóa học 2020 của Emmanuelle Charpentier và Jennifer Doudna, hoạt động như một công cụ chỉnh sửa văn bản cho hệ gen sinh học. Cơ chế gồm một RNA dẫn đường (guide RNA) dẫn dắt enzyme Cas9 đến đúng vị trí chuỗi nucleotide mục tiêu để cắt đứt liên kết đôi của DNA. Sau đó, cơ chế tự sửa chữa tế bào sẽ chèn hoặc xóa gen theo ý muốn, mở ra triển vọng chữa trị tận gốc các bệnh di truyền nan y như bệnh thiếu máu hồng cầu hình liềm và ung thư."""
    },

    # --- DOMAIN 6: Culture, History & Gastronomy ---
    {
        "id": 15,
        "title": "Nghệ Thuật Ẩm Thực Phở Việt Nam và Văn Hóa Nước Dùng Truyền Thống",
        "category": "Culture & Gastronomy",
        "tags": ["Ẩm Thực", "Phở", "Việt Nam", "Văn Hóa", "Gastronomy"],
        "summary": "Tinh hoa nước dùng ninh xương cùng hoa hồi, thảo quả và thảo mộc truyền thống Bắc Bộ.",
        "content": """Phở là biểu tượng văn hóa ẩm thực đặc trưng và tự hào nhất của Việt Nam trên bản đồ thế giới. Linh hồn của một bát phở truyền thống nằm ở nồi nước dùng trong vắt nhưng đậm đà, được ninh nhừ từ xương bò hoặc xương gà trong suốt 8 đến 12 tiếng đồng hồ cùng các gia vị tự nhiên nướng thơm như hành tím, gừng, hoa hồi, quế chi, thảo quả và đinh hương. Bánh phở mỏng dai làm từ bột gạo tươi kết hợp với thịt bò tái lăn, hành hoa thái nhỏ và ớt tươi tạo nên sự cân bằng ngũ hành hoàn mỹ trong ẩm thực Á Đông."""
    },
    {
        "id": 16,
        "title": "Quần Thể Di Sản Thế Giới Tràng An và Địa Mạo Karst Nhiệt Đới",
        "category": "Culture & Gastronomy",
        "tags": ["Tràng An", "Di Sản Thế Giới", "Du Lịch", "Ninh Bình", "Địa Mạo"],
        "summary": "Di sản hỗn hợp văn hóa và thiên nhiên độc nhất vô nhị của UNESCO tại Đông Nam Á.",
        "content": """Quần thể danh thắng Tràng An tại tỉnh Ninh Bình là di sản hỗn hợp đầu tiên của Đông Nam Á được UNESCO công nhận về cả giá trị văn hóa và thiên nhiên nổi bật toàn cầu. Nơi đây sở hữu cảnh quan địa mạo karst nhiệt đới nguyên sơ kỳ vĩ với những tháp đá vôi dựng đứng sừng sững bên dòng sông uốn lượn qua các hang động ngập nước. Các cuộc khai quật khảo cổ học tại hang Trống, hang Bói đã chứng minh dấu tích cư trú của người tiền sử thích nghi với sự biến đổi môi trường và khí hậu kéo dài hơn 30.000 năm."""
    },

    # --- DOMAIN 7: Economics & FinTech ---
    {
        "id": 17,
        "title": "Công Nghệ Sổ Cái Phân Tán Blockchain và Hợp Đồng Thông Minh",
        "category": "Economics & FinTech",
        "tags": ["Blockchain", "Smart Contracts", "Ethereum", "FinTech", "DeFi"],
        "summary": "Nguyên lý loại bỏ trung gian tài chính qua mạng ngang hàng và mật mã học bất đối xứng.",
        "content": """Blockchain là một cơ sở dữ liệu sổ cái phân tán không thể chỉnh sửa, nơi các giao dịch được nhóm thành từng khối (blocks) liên kết với nhau bằng mã băm mật mã học (cryptographic hash). Nền tảng Ethereum đã đưa blockchain lên một bước tiến mới với Hợp đồng Thông minh (Smart Contracts) - các đoạn mã chương trình tự động thực thi khi thỏa mãn các điều kiện định trước trên máy ảo EVM, đặt nền móng cho hệ sinh thái Tài chính Phi tập trung (DeFi), cho vay thế chấp và bảo hiểm tự động không cần ngân hàng trung gian."""
    },
    {
        "id": 18,
        "title": "Giao Dịch Tần Suất Cao (HFT) và Hạ Tầng Mạng Độ Trễ Cực Thấp",
        "category": "Economics & FinTech",
        "tags": ["HFT", "Trading", "FinTech", "Algorithms", "Low Latency"],
        "summary": "Ứng dụng thuật toán và vi mạch FPGA thực hiện giao dịch chứng khoán ở cấp độ micro-giây.",
        "content": """Giao dịch tần suất cao (High-Frequency Trading - HFT) là một hình thức giao dịch chứng khoán và phái sinh dựa trên các thuật toán tài chính tự động hóa cao độ. Các công ty HFT đặt máy chủ ngay tại trung tâm dữ liệu của các sàn giao dịch (Co-location) và sử dụng thẻ vi mạch FPGA/ASIC tùy biến cùng giao thức mạng không chạm Kernel (Kernel-Bypass) để đưa độ trễ khớp lệnh xuống mức nano-giây. Các chiến lược phổ biến bao gồm tạo lập thị trường (Market Making), kinh doanh chênh lệch giá (Statistical Arbitrage) và quét thanh khoản."""
    },
    {
        "id": 19,
        "title": "Hệ Thống Gợi Ý Hai Tháp (Two-Tower Model) Trong Thương Mại Điện Tử",
        "category": "AI & Deep Learning",
        "tags": ["Recommendation", "Two-Tower", "Retrieval", "HNSW", "E-Commerce"],
        "summary": "Tách biệt tháp người dùng và tháp vật phẩm để truy vấn thời gian thực hàng triệu sản phẩm.",
        "content": """Mô hình Hai Tháp (Two-Tower Neural Network) là kiến trúc công nghiệp chuẩn mực cho giai đoạn đề xuất ứng viên (Candidate Retrieval) tại YouTube, TikTok và Amazon. Tháp người dùng ánh xạ lịch sử hành vi, ngữ cảnh vào vector u, trong khi tháp vật phẩm ánh xạ thông tin sản phẩm vào vector v. Khi truy vấn thời gian thực, vector người dùng được đưa vào chỉ mục HNSW để tìm kiếm top K vật phẩm có tích vô hướng cực đại trong vòng chưa đầy 5 mili-giây giữa hàng trăm triệu sản phẩm."""
    },
    {
        "id": 20,
        "title": "DiskANN: Phá Vỡ Giới Hạn Bộ Nhớ RAM Cho Tìm Kiếm Vector Tỷ Điểm",
        "category": "Vector Search & Indexing",
        "tags": ["DiskANN", "Vamana", "SSD", "Large Scale", "Vector Index"],
        "summary": "Thuật toán đồ thị Vamana tối ưu hóa truy xuất ngẫu nhiên trên ổ cứng thể rắn SSD NVMe.",
        "content": """DiskANN của Microsoft Research giải quyết rào cản chi phí bộ nhớ đắt đỏ của HNSW khi số lượng vector đạt quy mô hàng tỷ. Khác với HNSW lưu trữ toàn bộ cấu trúc đồ thị trong RAM, DiskANN sử dụng đồ thị đơn tầng Vamana lưu trên ổ cứng SSD NVMe tốc độ cao kết hợp với nén vector Product Quantization (PQ) trong RAM để định tuyến. Bằng cách gộp các thao tác đọc đĩa (SSD block I/O), DiskANN đạt thông lượng hàng chục nghìn QPS với độ trễ thấp và chi phí phần cứng rẻ hơn 5 đến 10 lần."""
    },
    {
        "id": 21,
        "title": "Vắc-xin mRNA: Đột Phá Công Nghệ Sinh Học Phòng Ngừa Đại Dịch",
        "category": "Biomedicine & Genomics",
        "tags": ["mRNA", "Vaccine", "Biotechnology", "Immunology", "Medicine"],
        "summary": "Nguyên lý truyền đạt mã di truyền nhân tạo qua hạt nano lipid để kích hoạt miễn dịch.",
        "content": """Công nghệ vắc-xin mRNA đánh dấu bước nhảy vọt lịch sử của y học hiện đại, được vinh danh với giải Nobel Y sinh 2023 cho Katalin Karikó và Drew Weissman. Thay vì tiêm virus bất hoạt hoặc protein tái tổ hợp truyền thống, vắc-xin cung cấp chuỗi mRNA nhân tạo được đóng gói trong hạt nano lipid (LNP) hướng dẫn tế bào cơ thể tự tổng hợp protein gai (Spike protein) của mầm bệnh. Hệ miễn dịch sau đó tự nhận diện protein này và sản sinh kháng thể trung hòa cũng như tế bào T nhớ mà không gây nguy cơ nhiễm bệnh."""
    },
    {
        "id": 22,
        "title": "Mạng Lưới Chuyển Tiếp Tin Cậy (Zero-Knowledge Proofs - ZKP)",
        "category": "Economics & FinTech",
        "tags": ["ZKP", "zk-SNARKs", "Privacy", "Cryptography", "Blockchain"],
        "summary": "Phương pháp mật mã chứng minh tính đúng đắn của dữ liệu mà không làm lộ nội dung thông tin.",
        "content": """Bằng chứng không tiết lộ tri thức (Zero-Knowledge Proofs - ZKP), đặc biệt là zk-SNARKs và zk-STARKs, là công nghệ mật mã cho phép một bên (Prover) chứng minh với bên kia (Verifier) rằng một mệnh đề là đúng mà không cung cấp bất kỳ thông tin nào khác ngoài tính xác thực của mệnh đề đó. ZKP là nền tảng cốt lõi của các giải pháp mở rộng quy mô Layer 2 trên Ethereum (ZK-Rollups) và bảo mật danh tính số, cho phép kiểm tra tài chính hoặc độ tuổi hợp lệ mà không cần công khai số dư hay ngày sinh."""
    }
]


class DocumentStore:
    """
    Manages document corpus, metadata, persistence, and text formatting.
    """
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.documents: Dict[int, Dict[str, Any]] = {}
        self._load_initial_corpus()

    def _load_initial_corpus(self):
        if self.storage_path and os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    docs_list = json.load(f)
                    for doc in docs_list:
                        self.documents[doc["id"]] = doc
                print(f"[Corpus] Loaded {len(self.documents)} documents from '{self.storage_path}'")
                return
            except Exception as e:
                print(f"[Corpus Warning] Error reading custom storage ({e}). Using default corpus.")

        for doc in DEFAULT_DOCUMENTS:
            self.documents[doc["id"]] = doc
        print(f"[Corpus] Initialized with {len(self.documents)} default documents across 7 domains.")

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.documents.values())

    def get(self, doc_id: int) -> Optional[Dict[str, Any]]:
        return self.documents.get(doc_id)

    def add_document(self, title: str, content: str, category: str = "Tùy chỉnh", tags: Optional[List[str]] = None, summary: str = "") -> Dict[str, Any]:
        next_id = max(self.documents.keys(), default=-1) + 1
        if not summary:
            # Auto-generate summary from first sentence
            first_sentence = content.strip().split(".")[0]
            summary = (first_sentence[:120] + "...") if len(first_sentence) > 120 else first_sentence
        
        doc = {
            "id": next_id,
            "title": title.strip(),
            "category": category.strip(),
            "tags": tags if tags else ["Custom"],
            "summary": summary.strip(),
            "content": content.strip()
        }
        self.documents[next_id] = doc
        self.save()
        return doc

    def delete_document(self, doc_id: int) -> bool:
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.save()
            return True
        return False

    def save(self):
        if self.storage_path:
            try:
                with open(self.storage_path, "w", encoding="utf-8") as f:
                    json.dump(list(self.documents.values()), f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[Corpus Error] Could not save documents: {e}")

    def get_texts_for_embedding(self) -> List[Tuple[int, str]]:
        """
        Returns list of (doc_id, text_to_embed) combining title, tags, and content.
        """
        results = []
        for doc_id, doc in self.documents.items():
            tags_str = ", ".join(doc.get("tags", []))
            full_text = f"Title: {doc['title']}. Category: {doc['category']}. Tags: {tags_str}. Summary: {doc['summary']}. Content: {doc['content']}"
            results.append((doc_id, full_text))
        return results

    def highlight_snippet(self, content: str, query: str, max_chars: int = 240) -> str:
        """
        Generates an HTML snippet with highlighted query words.
        """
        words = [re.escape(w) for w in query.split() if len(w) > 2]
        if not words:
            return content[:max_chars] + "..." if len(content) > max_chars else content

        pattern = re.compile(r"(" + "|".join(words) + r")", re.IGNORECASE)
        match = pattern.search(content)
        
        if match:
            start_pos = max(0, match.start() - 60)
            end_pos = min(len(content), start_pos + max_chars)
            snippet = content[start_pos:end_pos]
            if start_pos > 0:
                snippet = "..." + snippet
            if end_pos < len(content):
                snippet = snippet + "..."
        else:
            snippet = content[:max_chars] + "..." if len(content) > max_chars else content

        # Highlight matches with <mark> tag
        highlighted = pattern.sub(r"<mark class='bg-amber-300 dark:bg-amber-600/60 text-slate-900 dark:text-white px-1 rounded font-medium'>\1</mark>", snippet)
        return highlighted
