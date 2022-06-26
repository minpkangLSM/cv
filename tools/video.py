import cv2

class videoReader :

    @staticmethod
    def videoReader(function,
                    **kwargs):

        while cv2.waitKey(1) < 0:
            ret, frame = kwargs['capture'].read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            feature = function(img=gray,
                               ksize=kwargs['ksize'],
                               sigmaX=kwargs['sigmaX'],
                               sigmaY=kwargs['sigmaY'],
                               thr=kwargs['thr'])
            cv2.imshow("Results", feature)

        kwargs['capture'].release()
        cv2.destroyWindow(0)
        return None
