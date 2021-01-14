# Lint as: python3
"""Using TF Lite to detect objects from camera."""
import argparse
import time
import os
from PIL import Image
from PIL import ImageDraw
import cv2
from Subfunctions import detect
import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
    }[platform.system()]


def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[tflite.load_delegate(
                EDGETPU_SHARED_LIB,{'device': device[0]} if device else {}
                )]
            )


def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),  '%s\n%.2f' % (labels[obj.label_id], obj.score),fill='red')

def append_objs_to_img(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape
    #print("height, width, channels: ",height, width, channels)
    for obj in objs:
        x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
        #print("Part1 x0, y0, x1, y1: ",x0, y0, x1, y1)
        #x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        #print("x0, y0, x1, y1: ",x0, y0, x1, y1)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels[obj.label_id])
        #print("label: ",label)
        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

def main():
  default_model_dir = './models'
  #default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
  default_model = 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
  default_labels = 'coco_labels.txt'
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', help='File path of .tflite file.', default=os.path.join(default_model_dir,default_model))
  parser.add_argument('-l', '--labels', help='File path of labels file.',default=os.path.join(default_model_dir,default_labels))
  parser.add_argument('-t', '--threshold', type=float, default=0.1, help='Score threshold for detected objects.')
  parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 1)
  args = parser.parse_args()

  labels = load_labels(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()
  #open camera
  cap = cv2.VideoCapture(args.camera_idx)
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    cv2_im = frame
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    #common.set_input(interpreter, pil_im)
    scale = detect.set_input(interpreter,pil_im.size,lambda size: pil_im.resize(size, Image.ANTIALIAS))
    interpreter.invoke()
    #print(scale)
    objs = detect.get_output(interpreter, args.threshold, scale)
    #print(objs)
    #draw_objects(ImageDraw.Draw(pil_im), objs, labels)

    cv2_im = append_objs_to_img(cv2_im, objs, labels)
    cv2.imshow('frame', cv2_im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()

  # image = Image.open(args.input)
  # scale = detect.set_input(interpreter,
    #                         image.size,
    #                         lambda size: image.resize(size, Image.ANTIALIAS)
    #                         )
    #
  # interpreter.invoke()
  # objs = detect.get_output(interpreter, args.threshold, scale)
  # # for obj in objs:
  # #   print(labels[obj.label_id])
  # #   print('  id: ', obj.id)
  # #   print('  score: ', obj.score)
  # #   print('  bbox: ', obj.bbox)
    #
  # if args.output:
  #   image = image.convert('RGB')
  #   draw_objects(ImageDraw.Draw(image), objs, labels)
  #   image.save(args.output)
  #   image.show()


if __name__ == '__main__':
  main()
