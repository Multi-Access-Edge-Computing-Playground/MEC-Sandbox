# import the necessary packages
from threading import Thread
import cv2
import socket
import struct
import pickle
import imutils
class Receive_Frame_And_Dict:
	def __init__(self):
		self.frame = None
		self.success = False
		self.temporary_dict_recv = {}
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
		self.conn=""

	def start(self,HOST,PORT):
		# start the thread to read frames from the video stream
		s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		print('GUI Socket created')
		s.bind((HOST,PORT))
		print('GUI Socket bind complete')
		s.listen(10)
		print('GUI Socket now listening')
		self.conn, address = s.accept()
		ip, port = str(address[0]), str(address[1])
		print("Connected with " + ip + ":" + port)
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		data = b""
		payload_size = struct.calcsize(">L")
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				self.conn.close()
				return
			# otherwise, read the next frame and button_dict from the stream
			#here we listen on port
			#receive image and button dict
			while len(data) < payload_size:
				data += self.conn.recv(2048*2)
			packed_msg_size = data[:payload_size]
			data = data[payload_size:]
			msg_size = struct.unpack(">L", packed_msg_size)[0]
			while len(data) < msg_size:
				data += self.conn.recv(2048*2)
			frame_data = data[:msg_size]
			data = data[msg_size:]
			frame_dict=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
			# frame_dict=pickle.loads(frame_data)
			frame=frame_dict["imageFile"]
			# print(frame)
			# d4d=frame_dict["depthFile"]
			self.temporary_dict_recv=dict(frame_dict["buttonDict"]) #change type because it was a multiprocess.dict
			self.frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
			self.success = True


	def read(self):
		# return the frame most recently read
		return self.success, self.frame, self.temporary_dict_recv

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

def main():
	# Parameters for receiving the image and button dictionary
	HOST=''
	PORT=8089
	# s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	# s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	# print('GUI Socket created')
	# s.bind((HOST,PORT))
	# print('GUI Socket bind complete')
	# s.listen(10)
	# print('GUI Socket now listening')
	# conn, address = s.accept()
	# ip, port = str(address[0]), str(address[1])
	# print("Connected with " + ip + ":" + port)
	"""#Combine:
	#put
	Thread(target=client_thread, args=(connection, ip, port)).start()
	#into"""
	vs = Receive_Frame_And_Dict().start(HOST,PORT)
	########
	# CONTINUE:
	# clientThread() must be  function of WebcamVideoStream
	# https://www.tutorialspoint.com/socket-programming-with-multi-threading-in-python

	# loop over some frames...this time using the threaded stream
	#
	print("starting to receive")
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		success, frame, temporary_dict_recv = vs.read()
		if success == True:
			# frame = imutils.resize(frame, width=400)
			# check to see if the frame should be displayed to our screen
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			if (key == 27) or (key == 13):
				break
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
if __name__ == '__main__':
	main()
