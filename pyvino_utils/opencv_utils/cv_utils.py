import cv2



def select_color(color):
    colors = {
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "white": (255, 255, 255),
    }
    return colors.get(color.lower(), 'green')
