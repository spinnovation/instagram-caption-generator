# 📸 Instagram 감상평 생성기

AI가 자동으로 Instagram 게시물의 감상평과 해시태그를 생성해주는 웹 애플리케이션입니다.

## 🌟 주요 기능

- **이미지 업로드**: PNG, JPG, JPEG 형식 지원
- **자동 크롭**: 얼굴 인식을 통한 4:5 비율 자동 크롭 (Instagram 최적화)
- **다양한 스타일**: 친근함, 감동적, 위트있는 톤 등 선택 가능
- **커스텀 프롬프트**: 직접 원하는 스타일 지정 가능
- **AI 감상평**: OpenAI GPT를 활용한 자연스러운 한국어 캡션 생성
- **해시태그 추천**: 관련성 높은 해시태그 자동 생성
- **비용 계산기**: API 사용 비용 실시간 계산 및 최적화 팁

## 🚀 라이브 데모

[여기서 바로 사용해보세요!](https://your-app-url.streamlit.app)

## 💻 로컬 실행 방법

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/instagram-caption-generator.git
cd instagram-caption-generator
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 실행
```bash
streamlit run instagram_caption_generator.py
```

### 4. 브라우저에서 접속
`http://localhost:8501`로 접속

## 🔑 API 키 설정

### 온라인 사용 (Streamlit Cloud)
- 웹 앱에서 직접 API 키를 입력하세요
- API 키는 브라우저 세션에만 저장되며 서버에 저장되지 않습니다

### 로컬 사용
다음 중 하나의 방법을 선택하세요:

**방법 1: 환경변수**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**방법 2: Streamlit secrets**
`.streamlit/secrets.toml` 파일 생성:
```toml
OPENAI_API_KEY = "your-api-key-here"
```

## 📋 OpenAI API 키 발급 방법

1. [OpenAI Platform](https://platform.openai.com) 접속
2. 계정 생성 또는 로그인
3. API Keys 메뉴에서 새 키 생성
4. 결제 방법 등록 (보통 $5 정도면 충분)

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **AI Model**: OpenAI GPT-4o-mini
- **Image Processing**: PIL, OpenCV
- **Face Detection**: OpenCV Haar Cascades
- **Language**: Python 3.8+

## 📁 프로젝트 구조

```
instagram-caption-generator/
├── instagram_caption_generator.py  # 메인 애플리케이션
├── requirements.txt               # 의존성 패키지
├── README.md                     # 프로젝트 설명
└── .gitignore                   # Git 무시 파일 목록
```

## 🎯 사용법

1. **API 키 입력**: OpenAI API 키를 입력하세요
2. **이미지 업로드**: 원하는 사진을 업로드하세요
3. **비용 확인**: 예상 API 사용 비용을 확인하세요
4. **스타일 선택**: 감상평의 톤앤매너를 선택하세요
5. **감상평 생성**: 버튼을 클릭하여 AI 감상평을 생성하세요
6. **복사 & 사용**: 생성된 텍스트를 Instagram에 사용하세요

## 💰 비용 정보

- **일반적인 사진 1장**: 약 $0.0001~0.0005 (0.13~0.65원)
- **고해상도 사진 1장**: 약 $0.0005~0.002 (0.65~2.6원)

실제로 매우 저렴하게 사용할 수 있습니다!

## 🔒 보안 및 개인정보

- API 키는 브라우저 세션에서만 사용됩니다
- 업로드된 이미지는 서버에 저장되지 않습니다
- 모든 처리는 실시간으로 이루어집니다

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 문의

문제가 있거나 제안사항이 있으시면 [Issues](https://github.com/your-username/instagram-caption-generator/issues)에 남겨주세요.

## 🙏 감사의 말

- [Streamlit](https://streamlit.io/) - 웹 앱 프레임워크
- [OpenAI](https://openai.com/) - AI 모델 제공
- [OpenCV](https://opencv.org/) - 이미지 처리