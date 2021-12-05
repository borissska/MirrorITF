import cv2
import mediapipe as mp
import time


class FaceMeshDetector:

    def __init__(self, staticMode=False, maxFaces=1, refLandmarks=False, minConf=0.5, minTrack=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refLandmarks = refLandmarks
        self.minConf = minConf
        self.minTrack = minTrack

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refLandmarks, self.minConf,
                                                 self.minTrack)
        self.drawSpec = self.mpDraw.DrawingSpec((0, 255, 0), thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLMS in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLMS, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLMS.landmark):
                    ih, iw, ic = img.shape
                    x, y, z = int(lm.x * iw), int(lm.y * ih), int(lm.z * ic)
                    # cv2.line(img, )168, 8,
                    if draw:
                        cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow('rez', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
