import cv2

# Video akışını başlat
cap = cv2.VideoCapture(0)

# Yüz tanıma için eğitilmiş OpenCV sınıflandırıcısını yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Kameradan görüntüyü oku
    ret, frame = cap.read()
    
    # Görüntüyü griye dönüştür (hız için)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Yüzleri algıla
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Algılanan yüzlerin etrafına dikdörtgen çiz
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # Yüz bölgesini kırp ve ayrı bir dosyaya kaydet
        face = frame[y:y+h, x:x+w]
        cv2.imwrite('face_detected.jpg', face)
        
    # Pencereye kareleri çizdir
    cv2.imshow('frame',frame)
    
    # 'q' tuşuna basılırsa döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video akışını durdur ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
