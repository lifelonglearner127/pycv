Color Based Tracking
====================
This examples track the object using color. Colored based tracking methods are very much suitable for real-time detection and tracking.


Ball Tracking
-------------
This python script we developed is able to 

- detect the presence of the colored ball,
- track and draw the position of the ball as it moved around the screen.

Running the script::

    python ball_tracking.py --video videos/ball_tracking.mp4


Ball Tracking Movement
----------------------
This python script is a simple extension to ball tracking example. This script is able to compute the moving direction of the ball.

Running the script::

    python ball_movement.py --video videos/ball_movement.mp4


Football Player Tracking
------------------------
There are multiple ways to detect players in any sports videos.
Here I have used simple image processing techniques to detect players by only using opencv.
It detects first the green ground and make everything other then green color into black.
After converting into greyscale I have found contours on the ground. By using some parameters we will detect players.
Here I have used the video of France VS Belgium match.
So for further detection, I have used the color of their jersy to segment them.
For france We will detect the blue jersy and then for belgium we will detect the red jersy.

Algorithm:

- First we will read the video. Detect the Green ground.
- Use morphological operation for better detection. Find contours. Detect players.
- Segment them by France or Belgium. Detect the football.

Running the script::

    python player_detect.py --video videos/match.mp4
