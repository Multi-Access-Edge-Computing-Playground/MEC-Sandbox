# Adding a watermark to multiple Videos at once
## Getting Started
1. Clone the repository with GitHub for Desktop using the link of this repository (https://github.com/Multi-Access-Edge-Computing-Playground/MEC-Sandbox.git)
2. Install [Python](https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe) (Version 3.7)
>> while walking through the install wizard make sure to enable PATH
![activate PATH](https://datatofish.com/wp-content/uploads/2018/10/0001_add_Python_to_Path.png)
3. Install some libraries by opening the Terminal (on Windows: search for cmd in the start-search) and execute the following commands:
```
pip install moviepy
```
6. Choose Videos and Logo:
put the video files that need a logo into the folder "Videos"
put the logo next to this file
open the file [wasserzeichen_add_logo_to_videos.py](wasserzeichen_add_logo_to_videos.py)
type the filename of the logo into the variable
```
logoname
```
like
```
logoname = "logo.png"
```
Also change parameters for the Logo Size and the margin of the bottom and right edge of the video
```
logo_height = 60 #pixels
logo_margin_right =  0 #pixels
logo_margin_bottom =  0 #pixels
```
7. Run the Program

>> The most basic and the easy way to run Python scripts is by using the python command. You need to open a command-line and cd into the folder containing the python file like:
```
cd C:\Users\fabian\Documents\GitHub\MEC-Sandbox\Add_Watermark_to_Videos
```
the type the word python followed by the name of your script file, like this:
```
python wasserzeichen_add_logo_to_videos.py
```

The script uses multiprocessing - the more cores, the faster!
The videos with the wartermarks will be located in the folder "Videos_Logo".
