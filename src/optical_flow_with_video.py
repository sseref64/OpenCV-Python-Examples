import cv2
import numpy as np

# Video dosyasını oku veya kamera akışını başlat
cap = cv2.VideoCapture('yourpath/video_name.mp4')  # Video dosyası adını buraya girin veya 0 olarak ayarlayarak kamera akışını kullanın

# Optik akış için önceki kareyi saklamak için değişkenler
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Optik akış parametreleri
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Rengi göstermek için rastgele renkler
color = np.random.randint(0, 255, (100, 3))

while True:
    # Yeni kareyi oku ve griye dönüştür
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Optik akış algoritmasını uygula
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Optik akış vektörlerini çiz
    for i, (x, y) in enumerate(flow.reshape(-1, 2)):
        if i % 5 == 0:  # Vektörlerin sıklığını ayarlamak için
            cv2.line(frame, (int(x), int(y)), (int(x)+1, int(y)+1), color[i].tolist(), 1)
    
    # Sonraki kare için önceki kareyi güncelle
    prev_gray = gray
    
    # Sonuçları göster
    cv2.imshow('Optical Flow', frame)
    
    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Video akışını durdur ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
