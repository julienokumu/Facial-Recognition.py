import cv2
import face_recognition

# Load known face encodings and names
known_face_encodings = []
known_face_names = []

#Load known faces and their names
known_person1_image = face_recognition.load_image_file("/home/kingsader/Desktop/Facial Recognition/Images/Christine.JPG")
known_person2_image = face_recognition.load_image_file("/home/kingsader/Desktop/Facial Recognition/Images/Gosling.JPG")
known_person3_image= face_recognition.load_image_file("/home/kingsader/Desktop/Facial Recognition/Images/Julien.JPG")
known_person4_image = face_recognition.load_image_file("/home/kingsader/Desktop/Facial Recognition/Images/Messi.JPG")

known_person1_encoding = face_recognition.face_encodings(known_person1_image)[0]
known_person2_encoding = face_recognition.face_encodings(known_person2_image)[0]
known_person3_encoding = face_recognition.face_encodings(known_person3_image)[0]
known_person4_encoding = face_recognition.face_encodings(known_person4_image)[0]

known_face_encodings.append(known_person1_encoding)
known_face_encodings.append(known_person2_encoding)
known_face_encodings.append(known_person3_encoding)
known_face_encodings.append(known_person4_encoding)

known_face_names.append("Christine Wambui")
known_face_names.append("Ryan Gosling")
known_face_names.append("Julien Okumu")
known_face_names.append("Lionel Messi")

#Initialize the Webcam
video_capture = cv2.VideoCapture(0)

while True:
    #Capture frame by frame
    ret, frame = video_capture.read()

    #Find all face locations in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    #Loop through each face found in the frame
    for (top, right, bottom, left), face_encodings in zip(face_locations, face_encodings):
        #Check if face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encodings)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index] 

        # Draw a box around the face and label with the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name,(left,top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0),2)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Break the loop when the key 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close Webcam and OpenCV windows
video_capture.release()   
cv2.destroyAllWindows()     