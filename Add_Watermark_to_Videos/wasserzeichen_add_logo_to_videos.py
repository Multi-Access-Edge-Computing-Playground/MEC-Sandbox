"""
Usage:
next to this file create two folders:
	folder1 named "Videos"
	folder2 named "Videos_Logo"
put the video files that need a logo into the folder "Videos"
put the logo next to this file
type the filename of the logo into the variable
logoname
like
logoname = "logo.png"
"""
logoname = "M40_BTU_Logo2.png"


from multiprocessing import Pool
import moviepy.editor as mp
import sys
import glob2
foldername="Videos"
video_list=glob2.glob("./"+foldername+"/*")
# for video_path in video_list:
# 	print("Adding Logo to ",video_path)
# 	video = mp.VideoFileClip(video_path)
#
# 	"""
# 	Change the parameters in this section to
# 		make the logo appear in the desired corner or
# 		change the size of the logo or
# 		add padding to move th elogo further into the image
# 	"""
# 	logo = (mp.ImageClip(logoname)
# 			  .set_duration(video.duration)
# 			  .resize(height=60) # if you need to resize...
# 			  .margin(right=0, bottom=0, opacity=0.5) # (optional) logo-border padding
# 			  .set_pos(("right","bottom")))
#
# 	final = mp.CompositeVideoClip([video, logo])
# 	final.write_videofile(video_path.replace(foldername,"Videos_Logo"),codec="libx264")
# 	#sys.exit()




def process_video(video_path):
	print("Adding Logo to ",video_path)
	video = mp.VideoFileClip(video_path)
	"""
	Change the parameters in this section to
		make the logo appear in the desired corner or
		change the size of the logo or
		add padding to move th elogo further into the image
	"""
	logo = (mp.ImageClip(logoname)
			  .set_duration(video.duration)
			  .resize(height=60) # if you need to resize...
			  .margin(right=0, bottom=0, opacity=0.5) # (optional) logo-border padding
			  .set_pos(("right","bottom")))

	final = mp.CompositeVideoClip([video, logo])
	final.write_videofile(video_path.replace(foldername,"Videos_Logo"),codec="libx264")

if __name__ == '__main__':
	pool = Pool()						 # Create a multiprocessing Pool
	pool.map(process_video, video_list)  # process data_inputs iterable with pool
