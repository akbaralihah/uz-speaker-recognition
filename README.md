# 🎙️ uz-speaker-recognition

Ushbu loyiha real vaqt rejimida o'zbek tilidagi audio xabarlarni matnga o'girish (STT), suhbatdoshlarni alohida ajratish (Diarization) va ovoz egasini aniqlash (Speaker Recognition) imkonini beruvchi sun'iy intellekt tizimidir. Loyiha FastAPI va WebSocket texnologiyalari asosida ishlaydi.

---

## ✨ Asosiy imkoniyatlar

- 👥 **Spikerlarni ajratish (Diarization):**  
  Audiodagi turli odamlar ovozini alohida segmentlarga bo'lish (masalan, `SPEAKER_00` va `SPEAKER_01`).

- 🗣 **Ovozdan tanish (Speaker Recognition):**  
  Oldindan ro'yxatdan o'tgan foydalanuvchilarni ovozidan tanib olish va ismini aniqlash.

- 📝 **O'zbek tili STT:**  
  Whisper modeli yordamida o'zbek tilidagi nutqni yuqori aniqlikda matnga aylantirish.

- ⚡ **Real vaqt tahlili:**  
  WebSocket orqali audio fayllarni uzluksiz qabul qilish va natijani darhol qaytarish.

---

## 🛠 Texnologiyalar (Barqaror versiyalar)

- **Backend:** FastAPI, Uvicorn, WebSockets  
- **AI Modellar:**  
  - Transformers (Whisper) – STT uchun  
  - Pyannote Audio – Diarization uchun  
  - SpeechBrain (ECAPA-TDNN) – Speaker Recognition uchun  
- **Audio ishlash:** Librosa, Torchaudio  

---

## 🚀 O'rnatish va ishga tushirish

O'z kompyuteringiz yoki serveringizda ishga tushirish uchun quyidagi qadamlarni bajaring:

### 1️⃣ Repozitoriyni klonlash (Clone)

```bash
git clone https://github.com/SIZNING_USERNAME/uz-speaker-recognition.git
cd uz-speaker-recognition
```

### 2️⃣ Virtual muhit yaratish va faollashtirish

Python'ning barqaror versiyasi (3.10) tavsiya etiladi:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac uchun
# venv\Scripts\activate   # Windows uchun
```

### 3️⃣ Kutubxonalarni o'rnatish

```bash
pip install -r requirements.txt
```

### 4️⃣ Muhit o'zgaruvchilarini sozlash (.env)

Loyiha papkasida `.env` faylini yarating va o'z kalitlaringizni kiriting:

```env
# .env fayl namunasi
PYANNOTATE_KEY=sizning_huggingface_tokeningiz_shu_yerga_yoziladi
```

> ⚠️ **Muhim:** Pyannote modelidan foydalanish uchun HuggingFace'da shartnomani qabul qilishingiz va Token yaratishingiz kerak.

### 5️⃣ Serverni ishga tushirish

```bash
python main.py
# Yoki to'g'ridan-to'g'ri Uvicorn orqali:
# uvicorn main:app --host 0.0.0.0 --port 8000
```

Server ishga tushgach, API hujjatlarini ko'rish uchun brauzerda kirishingiz mumkin:  
```
http://127.0.0.1:8000/docs
```

---

## 📡 Qanday ishlatiladi?

### 1️⃣ Spikerni ro'yxatdan o'tkazish

Ovoz orqali ismni tanish uchun avval tizimga audioni kiritish kerak.

- **Endpoint:** `POST /register-speaker`  
- **Parametrlar:**  
  - `full_name` (Matn)  
  - `file` (WAV/OGG audio fayl)

---

### 2️⃣ WebSocket orqali tahlil qilish (Client)

Audioni serverga yuborib, tahlil natijalarini olish uchun tayyor `client.py` skriptidan foydalaning:

```bash
python client.py
```

Skript sizdan audio fayl raqamini so'raydi. U websocket orqali serverga ulanadi va real vaqtda quyidagicha natija beradi:

```text
[0.7s - 2.8s] Akbarali (SPEAKER_01): salom ulug'bek, qalaysan, ishlar qalay?
[3.1s - 5.7s] Ulug'bek (SPEAKER_00): salom, yaxshi rahmat, o'zingchi bandmisan bugun.
```

---

## 🤝 Hissa qo'shish (Contributing)

Agar loyihani rivojlantirishga hissa qo'shmoqchi bo'lsangiz, Pull Request yuborishingiz yoki topilgan xatoliklarni Issues bo'limida qoldirishingiz mumkin.

