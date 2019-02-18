import cv2 as cv
import numpy as np
import random
import math
import sys
from pydub import AudioSegment as seg

# setting up the input file
file_name = 'monkey.mov'
back_name = 'background/background.mp4'

demo_video = cv.VideoCapture(file_name)
back_video = cv.VideoCapture(back_name)

# textures
replace_img = cv.imread('textures/replace.jpeg', 1)
original_h, original_w = np.size(replace_img, 0), np.size(replace_img, 1)

feather_height, feather_width = 35, 12

feather1 = cv.imread('textures/feather1.jpeg', 1)
feather1 = cv.resize(feather1, (feather_width, feather_height))

feather2 = cv.imread('textures/feather2.jpeg', 1)
feather2 = cv.resize(feather2, (feather_width, feather_height))

feathers = [feather1, feather2]

plane = cv.imread('textures/plane.jpg', 1)
plane = cv.resize(plane, (30, 30))

# sound files
magic_ball_sound = seg.from_wav('sound/magicball.wav')
eagle_sound = seg.from_wav('sound/eagle.wav')
thunder_sound = seg.from_wav('sound/thunder.wav')
background_sound = seg.silent(duration=32000)
milli_sec_per_frame = 1000/30

thunder_duration = 10

# attributes of the file
width = int(demo_video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(demo_video.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (width, height)
# motion estimation settings
frame_rate = demo_video.get(cv.CAP_PROP_FPS)

fourcc = cv.VideoWriter_fourcc('F', 'M', 'P', '4')


# setting up output file
out = cv.VideoWriter('output.avi', fourcc, 30, size)

print("Start, configures: ")
print("height " + str(height) + " width: " + str(width))


class Feather:
    def __init__(self, h, w, c):
        self.h = h + feather_height/2
        self.w = w
        self.colour = c

    def render(self):
        self.h += 3
        self.w += random.randint(-2, 2)
        if self.h > height or self.h - feather_height < 0:
            del self
        elif self.w + feather_width/2 > width or self.w - feather_width/2 < 0:
            del self


class Plane:
    def __init__(self):
        self.h = -15
        self.w = random.randint(15, width-15)

    def render(self):
        self.h += 4
        if self.h > height+15:
            del self


class MagicPoint:
    def __init__(self):
        self.h = height+10
        self.w = random.randint(15, width-15)
        self.v_h = 1
        self.v_w = random.randint(-1, 1)
        self.colour = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    def render(self, target):
        h_vel = belly[0]-self.h
        w_vel = belly[1] - self.w
        n = math.sqrt(math.pow(h_vel, 2) + math.pow(w_vel, 2))
        self.v_h = int(5*h_vel/n)
        self.v_w = int(5*w_vel/n)
        self.h += self.v_h
        self.w += self.v_w


# limit the number n between min_n to max_n


def clamp(n, min_n, max_n):
    return int(max(min(max_n, n), min_n))


# find the distance between two coordinates


def des(c1, c2):
    sq1 = math.pow(c1[0]-c2[0], 2)
    sq2 = math.pow(c1[1] - c2[1], 2)
    return math.sqrt(sq1 + sq2)


# find a motion point


def check_motion_point(image, motion_points, h, w):
    detect_range = 35
    for p in motion_points:
        if p[0]+detect_range > h > p[0]-detect_range and p[1]+detect_range > w > p[1]-detect_range:
            return motion_points
    bot = 0
    right = 0
    top = sys.maxsize
    left = sys.maxsize
    count = 0
    total = 0
    scan_range = 4
    for i in range(clamp(h-scan_range, 0, height), clamp(h+scan_range+1, 0, height)):
        for j in range(clamp(w-scan_range, 0, width), clamp(w+scan_range+1, 0, width)):
            total += 1
            if image[i][j][2] == 255:
                count += 1
                top = min(i, top)
                bot = max(i, bot)
                left = min(j, left)
                right = max(j, right)
    if count > total/1.4:
        new_point = ((bot+top)/2, (left+right)/2)
        motion_points.append(new_point)
    return motion_points


# gets the position of different body components
# takes all motions points as input
# output an array of positions for feet, an array of positions of hands, and one for belly


def split_motion_points(motion_points):
    feet = motion_points[-2:]
    hands = []
    belly = []
    left = sys.maxsize
    right = 0
    for p in motion_points[:-2]:
        left = min(p[1], left)
        right = max(p[1], right)
    for p in motion_points[:-2]:
        if left == p[1]:
            hands.append(p)
        if right == p[1]:
            hands.append(p)
        else:
            belly.append(p)
    belly_h = 0
    belly_w = 0
    belly_pos = (0, 0)
    if len(belly) > 0:
        for b in belly:
            belly_h += b[0]
            belly_w += b[1]
        belly_pos = (belly_h/len(belly), belly_w/len(belly))
    else:
        hands_ave_height = 0
        hands_ave_width = 0
        feet_ave_height = 0
        feet_ave_width = 0
        for h in hands:
            hands_ave_height += h[0]
            hands_ave_width += h[1]
        if len(hands) > 0:
            hands_ave_height /= len(hands)
            hands_ave_width /= len(hands)
        for f in feet:
            feet_ave_height += f[0]
            feet_ave_width += f[1]
        if len(feet) > 0:
            feet_ave_height /= len(feet)
            feet_ave_width /= len(feet)

        belly_h = int((hands_ave_height+feet_ave_height)/2)
        belly_w = int((hands_ave_width+feet_ave_width)/2)
        belly_pos = (belly_h, belly_w)

        if len(hands) == 1:
            hands.append((hands[1][0], 2*belly_w-hands[1][1]))
        if len(hands) == 0:
            hands = [(belly_h, belly_w - 30), (belly_h, belly_w + 30)]

    return hands, feet, belly_pos
# start processing


count = 0

back_ret, back_frame = back_video.read()
next_back_frame = None

magic_point_array = []
feather_array = []
plane_array = []
collide = False
pre_pos = (int(height/2), int(width/2))

colour = (0, 0, 0)
change_colour = False

thunder_time = 0

while True:
    motion_points = []
    ret, frame = demo_video.read()
    back_ret, back_frame = back_video.read()
    if not ret or not back_ret:
        break
    back_frame = cv.resize(back_frame, (width, height))
    out_image = np.zeros((height, width, 3), np.uint8)
    marked_image = np.zeros((height, width, 3), np.uint8)
    # go through every pixel of the frame and the frame after
    # when there is a detection need to be done, it finds the most similar block in next frame
    # if the colour difference or distance of moved block reaches threshold, draw a white dot on the graph at that pixel
    # so the parts of the video that are moving will be highlighted with white dots
    for h in range(0, height):
        for w in range(0, width):
            # Green
            condition_G = frame[h][w][1] < 200
            # Red
            condition_R = frame[h][w][2] > 180
            if condition_G and condition_R:
                marked_image[h][w] = (0, 0, 255)
    for h in range(0, height):
        for w in range(0, width):
            if marked_image[h][w][2] == 255:
                motion_points = check_motion_point(marked_image, motion_points, h, w)
    if thunder_time > 0:
        thunder_time -= 1
        for h in range(0, height):
            for w in range(0, width):
                for i in range(3):
                    back_frame[h][w][i] = clamp(back_frame[h][w][i]*2, 0, 255)
    else:
        die = random.randint(1, 50)
        if die == 5:
            thunder_time = thunder_duration

    out_image = back_frame
    # get coordinates for different body components
    hands, feet, belly = split_motion_points(motion_points)
    # draw circles on positions of hands, feet and belly

    #for h in hands:
        #cv.circle(out_image, (h[1], h[0]), 10, (0, 0, 255))
        #cv.line(out_image, (h[1], h[0]), (belly[1], belly[0]), (0, 0, 255))
    #for f in feet:
        #cv.circle(out_image, (f[1], f[0]), 10, (0, 0, 255))
        #cv.line(out_image, (f[1], f[0]), (belly[1], belly[0]), (0, 0, 255))
    #cv.circle(out_image, (belly[1], belly[0]), 10, (0, 0, 255))

    # resize the bird image based on the monkey
    monkey_width = int(clamp(math.fabs(hands[0][1]-hands[1][1]), 40, 300))
    ratio = float(monkey_width)/original_w
    changed_height = int(original_h*ratio)
    replace_img_resize = cv.resize(replace_img, (monkey_width, changed_height))
    img_h, img_w = np.size(replace_img_resize, 0), np.size(replace_img_resize, 1)

    # draws the bird on the position of the marionette
    height_replace_range = (clamp(belly[0]-img_h/2, 0, height), clamp(belly[0]+img_h/2, 0, height))
    width_replace_range = (clamp(belly[1]-img_w/2, 0, width), clamp(belly[1]+img_w/2, 0, width))
    for h in range(height_replace_range[0], height_replace_range[1]):
        for w in range(width_replace_range[0], width_replace_range[1]):
            condition_1 = replace_img_resize[h-height_replace_range[0]][w-width_replace_range[0]][0] < 100
            condition_2 = replace_img_resize[h-height_replace_range[0]][w-width_replace_range[0]][1] < 100
            condition_3 = replace_img_resize[h-height_replace_range[0]][w-width_replace_range[0]][2] < 100
            if condition_1 and condition_2 and condition_3:
                out_image[h][w] = colour

    # add a magic point per second
    if count % 30 == 0:
        new_magic_point = MagicPoint()
        magic_point_array.append(new_magic_point)
    # render magic points
    for m in magic_point_array:
        m.render(belly)
        cv.line(out_image, (m.w, m.h), (m.w, m.h), m.colour, thickness=10)
        if belly[0]+20 > m.h > belly[0]-20 and belly[1]+20 > m.w > belly[1]-20:
            colour = m.colour
            magic_point_array.remove(m)
            change_colour = True

    # add two planes per second
    if count % 10 == 0:
        new_plane = Plane()
        plane_array.append(new_plane)

    # render and draw plane
    for p in plane_array:
        if p:
            p.render()
        if p:
            index = random.randint(0, 1)
            for h in range(clamp(p.h - 15, 0, height), clamp(p.h + 15, 0, height)):
                for w in range(clamp(p.w - 15, 0, width), clamp(p.w + 15, 0, width)):
                    if plane[h - p.h + 15][w - p.w + 15][0] < 100:
                        out_image[h][w] = plane[h - p.h + 15][w - p.w + 15]

    # check if any plane hits the bird
    for p in plane_array:
        if p and belly[0]+img_h/2-10 > p.h > belly[0]-img_h/2+10 and belly[1]+img_w/2-10 > p.w > belly[1]-img_w/2+10:
            collide = True
            plane_array.remove(p)

    # if the movement of the bird is big, feathers fell down
    if collide:
        for n in range(3 + random.randint(0, 3)):
            new_feather = Feather(belly[0] + random.randint(-50, 50), belly[1] + random.randint(-30, 30), colour)
            feather_array.append(new_feather)
    # draw feathers
    for f in feather_array:
        if f:
            f.render()
        if f:
            index = random.randint(0, 1)
            feather_img = feather1
            if index == 1:
                feather_img = feather2
            for h in range(clamp(f.h - feather_height/2, 0, height), clamp(f.h + feather_height/2, 0, height)):
                for w in range(clamp(f.w - feather_width/2, 0, width), clamp(f.w + feather_width/2, 0, width)):
                    if feather_img[int(h - f.h + feather_height/2)][int(w - f.w + feather_width/2)][0] < 100:
                        out_image[h][w] = f.colour

    # sound track
    if collide:
        sec = int(count * milli_sec_per_frame)
        background_sound = background_sound.overlay(eagle_sound, position=sec)
        collide = False
    if thunder_time == thunder_duration:
        sec = int(count * milli_sec_per_frame)
        background_sound = background_sound.overlay(thunder_sound, position=sec)
    if change_colour:
        sec = int(count * milli_sec_per_frame)
        background_sound = background_sound.overlay(magic_ball_sound, position=sec)
        change_colour = False

    pre_pos = belly

    out.write(out_image)
    count += 1
    print("frame " + str(count) + " done")


background_sound.export('soundtrack.wav', format='wav')

out.release()
demo_video.release()
cv.destroyAllWindows()
print("Process finished")
