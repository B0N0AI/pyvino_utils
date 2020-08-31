import cv2
import matplotlib.pyplot as plt


def select_color(color):
    colors = {
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "white": (255, 255, 255),
    }
    return colors.get(color.lower(), (0, 255, 0))


def plot_current_frame(image):
    """Helper function for finding image coordinates/px"""
    img = image[:, :, 0]
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def add_text_to_image(
    image, position, text="", font_size=0.75, color="white", thickness=1
):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_COMPLEX,
        font_size,
        select_color(color),
        thickness,
    )


# WIP
class Contours:
    _first_frame = None

    def get_contours(self, gray_p_frame):
        """
        Ref: https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

        Motion tracking: https://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/
        """
        # if the first frame is None, initialize it
        if self._first_frame is None:
            self._first_frame = gray_p_frame

        # compute the absolute difference between the current frame and
        # first frame
        frame_delta = cv2.absdiff(self._first_frame, gray_p_frame)
        _, thresh = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)

        # dilate the threshold image to fill in holes, then find contours
        # on threshold image
        # Taking a matrix of size 10 as the kernel
        kernel = np.ones((5, 5), np.uint8)
        # kernel = None

        # thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.dilate(thresh, kernel, iterations=3)
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # TODO:
        # Add logic for tracking movements...See get_contours() docstring for some ideas on
        # adding the logic, also get better lighting video
        if contours:
            for cnt in contours:
                if cv2.contourArea(cnt) < 1000:
                    continue

                M = cv2.moments(cnt)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # cv2.drawContours(image, cnt, -1, (0, 0, 255), 5)
                y, x = image.shape[:2]
                cv2.circle(image, (x // 2, y // 2), 200, (0, 255, 0), 20)

                cv2.circle(image, (cx, cy), 7, (0, 255, 0), -1)
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (xmin, ymin, xmax, ymax) = cv2.boundingRect(cnt)

                cv2.rectangle(
                    image, (xmin, ymin), (xmin + xmax, ymin + ymax), (255, 255, 0), 2
                )

                # rectagleCenterPont = ((2 * xmin + xmax) // 2, (2 * ymin + ymax) // 2)

                # cv2.circle(image, rectagleCenterPont, 1, (0, 255, 255), 20)

            # for cnt in contours:
            #     # if the contour is too small, ignore it
            #     if cv2.contourArea(cnt) < 12000:
            #         continue
            #     cv2.line(
            #         image, (400,0), (400,0), (250, 0, 1), 2,
            # )  # blue line
            # cv2.line(
            #     frame, 800, y1, (0, 0, 255), 2,
            # )  # red line

            # # compute the bounding box for the contour, draw it on the frame,
            # # and update the text
            # (xmin, ymin, xmax, ymax) = cv2.boundingRect(cnt)

            # cv2.rectangle(
            #     frame, (xmin, ymin), (xmin + xmax, ymin + ymax), (0, 255, 0), 2
            # )

            # rectagleCenterPont = ((2 * xmin + xmax) // 2, (2 * ymin + ymax) // 2)

            # cv2.circle(frame, rectagleCenterPont, 1, (0, 0, 255), 1)

            # if testIntersectionIn(xmin, ymax):
            #     textIn += 1

            # if testIntersectionOut(xmin, ymax):
            #     textOut += 1
