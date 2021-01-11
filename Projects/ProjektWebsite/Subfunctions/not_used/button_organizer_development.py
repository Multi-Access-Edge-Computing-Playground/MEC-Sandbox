import cv2
import numpy as np
# # Follow Button
# cv2.rectangle(rgb_frame,(10,45),(90,75),(255,255,0),1)
# cv2.putText(rgb_frame, "Follow", (25,70),fontStyle, fontScale, (255,255,0), fontThickness)
# temporary_dict["Follow"] = {
#     "rectangle" : [10,45,90,75],
#     "on_click" : "Follow",
#     }
# # Inspect Button
# cv2.rectangle(rgb_frame,(10,80),(90,110),(255,255,0),1)
# cv2.putText(rgb_frame, "Inspect", (25,105),fontStyle, fontScale, (255,255,0), fontThickness)
# temporary_dict["Inspect"] = {
#     "rectangle" : [10,90,80,110],
#     "on_click" : "Inspect",
#     }
# # Click Mode Button
# cv2.rectangle(rgb_frame,(10,80+35),(90,110+35),(255,255,0),1)
# cv2.putText(rgb_frame, "Tap2Touch", (25,105+35),fontStyle, fontScale, (255,255,0), fontThickness)
# temporary_dict["Tap2Touch"] = {
#     "rectangle" : [10,90,80+35,110+35],
#     "on_click" : "Tap2Touch",
#     }
# improve Buttons with cv2.getTextSize(text, font, font_scale, thickness)
class Button():
    def __init__(self):
        self.fontStyle = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.7
        self.fontThickness = 2
        self.color = (255,255,0)
        self.button_px_delta_y = 35 #(depends on font)
        self.x_padding = 10
        self.y_padding = 5
        self.x_bbox_padding = 10
        self.y_bbox_padding = 10

    def insert_button(self,frame,temporary_dict,text,position_id=0,corner="top_left"):
        color=self.color
        fontStyle=self.fontStyle
        fontScale=self.fontScale
        fontThickness=self.fontThickness

        (text_width, text_height), baseline = cv2.getTextSize(text, fontStyle, fontScale, fontThickness)
        if corner=="top_left":
            corner_offset=[0,0]
        elif corner=="top_right":
            height, width = frame.shape[:2]
            corner_offset=[width-(self.x_padding+self.x_bbox_padding+text_width+self.x_bbox_padding+self.x_padding) ,0]
        elif corner=="bottom_left":
            height, width = frame.shape[:2]
            corner_offset=[0,height-(position_id*self.y_padding+position_id*self.button_px_delta_y +2*self.button_px_delta_y+position_id*self.button_px_delta_y+position_id*self.y_padding+self.y_padding)]
        elif corner=="bottom_right":
            height, width = frame.shape[:2]
            corner_offset=[ width-(self.x_padding+self.x_bbox_padding+text_width+self.x_bbox_padding+self.x_padding) ,height-(position_id*self.y_padding+position_id*self.button_px_delta_y +2*self.button_px_delta_y+position_id*self.button_px_delta_y+position_id*self.y_padding+self.y_padding)]
        #bounding box dimensions
        # print(corner_offset)
        x1_bb = corner_offset[0]+ self.x_padding
        y1_bb = corner_offset[1]+ position_id*self.y_padding+position_id*self.button_px_delta_y +self.button_px_delta_y
        x2_bb = corner_offset[0]+ self.x_padding+self.x_bbox_padding+text_width+self.x_bbox_padding
        # y2_bb = corner_offset[1]+ position_id*self.y_padding+position_id*self.button_px_delta_y +2*self.button_px_delta_y
        y2_bb = y1_bb + self.button_px_delta_y
        #text start coordinate
        x1_tx, y1_tx =x1_bb+self.x_bbox_padding,y2_bb-self.y_bbox_padding


        cv2.rectangle(frame,(x1_bb,y1_bb),(x2_bb,y2_bb),color,1)
        cv2.putText(frame, text, (x1_tx,y1_tx),fontStyle, fontScale, color, fontThickness)
        temporary_dict[text] = {
            "rectangle" : [x1_bb,x2_bb,y1_bb,y2_bb],
            "on_click" : text,
            }
        return frame, temporary_dict
#
#
# (text_width, text_height), baseline = cv2.getTextSize(class_names[cls],
#                                                               cv2.FONT_HERSHEY_SIMPLEX,
#                                                               0.75, 1)
#         cv2.rectangle(frame,
#                       (xy[0], xy[1]),
#                       (xy[0] + test_width, xy[1] - text_height - baseline),
#                       color[::-1],
#                       thickness=cv2.FILLED)
#         cv2.putText(frame, class_names[cls], (xy[0], xy[1] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
def main():
    height,width=600,1000
    blank_image = np.zeros((height,width,3), np.uint8)
    temporary_dict={}
    ButtonGenerator = Button()

    blank_image, temporary_dict = ButtonGenerator.insert_button(
                blank_image, temporary_dict, "Hello this is the first button!",position_id=0,corner="top_left")
    blank_image, temporary_dict = ButtonGenerator.insert_button(
                blank_image, temporary_dict, "Hello this is the 2nd button!",position_id=1,corner="top_left")
    blank_image, temporary_dict = ButtonGenerator.insert_button(
                blank_image, temporary_dict, "Hello",position_id=2,corner="top_left")

    blank_image, temporary_dict = ButtonGenerator.insert_button(
                blank_image, temporary_dict, "Hello this is the first button!",position_id=0,corner="top_right")
    blank_image, temporary_dict = ButtonGenerator.insert_button(
                blank_image, temporary_dict, "Hello this is the 2nd button!",position_id=1,corner="top_right")
    blank_image, temporary_dict = ButtonGenerator.insert_button(
                blank_image, temporary_dict, "Hello",position_id=2,corner="top_right")

    blank_image, temporary_dict = ButtonGenerator.insert_button(
                blank_image, temporary_dict, "Hello this is the first button!",position_id=0,corner="bottom_left")
    blank_image, temporary_dict = ButtonGenerator.insert_button(
                blank_image, temporary_dict, "Hello this is the 2nd button!",position_id=1,corner="bottom_left")
    blank_image, temporary_dict = ButtonGenerator.insert_button(
                blank_image, temporary_dict, "Hello",position_id=2,corner="bottom_left")

    blank_image, temporary_dict = ButtonGenerator.insert_button(
                blank_image, temporary_dict, "Hello this is the first button!",position_id=0,corner="bottom_right")
    blank_image, temporary_dict = ButtonGenerator.insert_button(
                blank_image, temporary_dict, "Hello this is the 2nd button!",position_id=1,corner="bottom_right")
    blank_image, temporary_dict = ButtonGenerator.insert_button(
                blank_image, temporary_dict, "Hello",position_id=2,corner="bottom_right")
    cv2.imshow("test", blank_image)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    # print("received a frame!")
    print(temporary_dict)

if __name__ == '__main__':
    main()
