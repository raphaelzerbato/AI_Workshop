### 🚀 Quick Start with Docker

```bash
git clone https://github.com/yourusername/your-workshop.git
cd your-workshop
docker build -t workshop-env .
docker run -p 8888:8888 -v $(pwd):/app workshop-env