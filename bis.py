import cv2 as cv
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):

    image = cv.flip(cv.imread(file), 1)
    results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv.flip(annotated_image, 1))

    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

cap = cv.VideoCapture(0)
fire_cascade=cv.CascadeClassifier("fire_detection.xml")
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    
    
    if not success:
      print("Ignoring empty camera frame.")
      continue
  
    gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    fire=fire_cascade.detectMultiScale(image,1.2,5)
    for (x,y,w,h) in fire:
        cv.rectangle(image,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)      
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w] 
        print("fire is detected")

    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    
    
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        x9,y9=hand_landmarks.landmark[9].x,hand_landmarks.landmark[9].y
        x12,y12=hand_landmarks.landmark[12].x,hand_landmarks.landmark[12].y
        x4,y4=hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y
        x1,y1=hand_landmarks.landmark[1].x,hand_landmarks.landmark[1].y
        x8,y8=hand_landmarks.landmark[8].x,hand_landmarks.landmark[8].y
        x5,y5=hand_landmarks.landmark[5].x,hand_landmarks.landmark[5].y
        x16,y16=hand_landmarks.landmark[16].x,hand_landmarks.landmark[16].y
        x13,y13=hand_landmarks.landmark[13].x,hand_landmarks.landmark[13].y
        x20,y20=hand_landmarks.landmark[20].x,hand_landmarks.landmark[20].y
        x17,y17=hand_landmarks.landmark[17].x,hand_landmarks.landmark[17].y
        if (y9<y12 )& (y5<y8) &(y13<y16)& (y17<y20):
            print("kapalı")
       
       
    # Flip the image horizontally for a selfie-view display.
    cv.imshow('MediaPipe Hands', cv.flip(image, 1))
    if cv.waitKey(5) & 0xFF == 27:
      break

cap.release()

cv.destroyAllWindows




