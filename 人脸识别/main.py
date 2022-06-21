import sensor
import image
import lcd
import KPU as kpu
import time
from Maix import FPIOA, GPIO
import gc
from fpioa_manager import fm
from board import board_info
import utime
import os
import ubinascii
from machine import UART

task_fd = kpu.load("/sd/task_fd.smodel") # 加载人脸检测模型
task_ld = kpu.load("/sd/task_ld.smodel") # 加载人脸五点关键点检测模型
task_fe = kpu.load("/sd/task_fe.smodel") # 加载人脸196维特征值模型

clock = time.clock()

#=================按键===================#
fm.register(board_info.BOOT_KEY, fm.fpioa.GPIOHS0)
key_gpio = GPIO(GPIO.GPIOHS0, GPIO.IN)
start_processing = False

fm.register(1, fm.fpioa.GPIOHS13)
keypad_gpio = GPIO(GPIO.GPIOHS13, GPIO.IN)
key_left = False

BOUNCE_PROTECTION = 50

def set_key_state(*_):
    global start_processing
    start_processing = True
    utime.sleep_ms(BOUNCE_PROTECTION)

def set_key_state_left(*_):
    global key_left
    key_left = True
    utime.sleep_ms(BOUNCE_PROTECTION)

key_gpio.irq(set_key_state, GPIO.IRQ_RISING, GPIO.WAKEUP_NOT_SUPPORT)
keypad_gpio.irq(set_key_state_left, GPIO.IRQ_RISING, GPIO.WAKEUP_NOT_SUPPORT)

lcd.init()
sensor.reset()
lcd.rotation(2)
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_hmirror(1)
sensor.set_vflip(1)
sensor.run(1)
anchor = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437,6.92275, 6.718375, 9.01025)  # anchor for face detect
dst_point = [(44, 59), (84, 59), (64, 82), (47, 105),(81, 105)]  # standard face key point position
a = kpu.init_yolo2(task_fd, 0.5, 0.3, 5, anchor)
img_lcd = image.Image()
img_face = image.Image(size=(128, 128))
a = img_face.pix_to_ai()

#=================内存卡===================#
feature_file_exists = 0
for v in os.ilistdir('/sd'):#to check key directorys or files in sd card.sd card should be formated to fat32
    if v[0] == 'features.txt' and v[1] == 0x8000:#0x8000 is file
        feature_file_exists = 1
#================内存卡读写操作================#
record_ftr=[] #空列表 用于存储当前196维特征
record_ftrs=[] #空列表 用于存储按键记录下人脸特征， 可以将特征以txt等文件形式保存到sd卡后，读取到此列表，即可实现人脸断电存储。
names = ['Pang_sen_lin', 'Bai_jing_ting', 'Jiang_wen', 'Peng_yu_yan', 'Zhao_li_ying',  'Jamie_Foxx', 'member.1', 'member.2'] # 人名标签，与上面列表特征值一一对应。

reco = ''
record = []
def save_feature(feat):
    with open('/sd/features.txt','a') as f:
        record =ubinascii.b2a_base64(feat)
        f.write(record)

st = ''
if(feature_file_exists):
    print("start")
    with open('/sd/features.txt','rb') as f:
        s = f.readlines()
        print(len(s))
        for line in s:
            record_ftrs.append(bytearray(ubinascii.a2b_base64(line)))

#=================串口===================#
fm.register(13, fm.fpioa.UART1_TX, force=True)
fm.register(12, fm.fpioa.UART1_RX, force=True)

uart = UART(UART.UART1, 115200, 8, None, 1, timeout=50, read_buf_len=4096)

#=================人脸识别===================#
ACCURACY = 65
flag = -1
record_1 = 0
record_2 = 0


while(True):
    img = sensor.snapshot()
    clock.tick()

    read_data = uart.read()
    if read_data == b'A':
       key_left = True
       read_data = b'Z'
    elif read_data == b'B':
       start_processing = True
       read_data = b'Z'

    if key_left:

        code = kpu.run_yolo2(task_fd, img)
        if code:
            for i in code:
                # Cut face and resize to 128x128
                a = img.draw_rectangle(i.rect())
                face_cut = img.cut(i.x(), i.y(), i.w(), i.h())
                face_cut_128 = face_cut.resize(128, 128)
                a = face_cut_128.pix_to_ai()
                # a = img.draw_image(face_cut_128, (0,0))
                # Landmark for face 5 points
                fmap = kpu.forward(task_ld, face_cut_128)
                plist = fmap[:]
                le = (i.x() + int(plist[0] * i.w() - 10), i.y() + int(plist[1] * i.h()))
                re = (i.x() + int(plist[2] * i.w()), i.y() + int(plist[3] * i.h()))
                nose = (i.x() + int(plist[4] * i.w()), i.y() + int(plist[5] * i.h()))
                lm = (i.x() + int(plist[6] * i.w()), i.y() + int(plist[7] * i.h()))
                rm = (i.x() + int(plist[8] * i.w()), i.y() + int(plist[9] * i.h()))
                a = img.draw_circle(le[0], le[1], 4)
                a = img.draw_circle(re[0], re[1], 4)
                a = img.draw_circle(nose[0], nose[1], 4)
                a = img.draw_circle(lm[0], lm[1], 4)
                a = img.draw_circle(rm[0], rm[1], 4)
                # align face to standard position
                src_point = [le, re, nose, lm, rm]
                T = image.get_affine_transform(src_point, dst_point)
                a = image.warp_affine_ai(img, img_face, T)
                a = img_face.ai_to_pix()
                # a = img.draw_image(img_face, (128,0))
                del (face_cut_128)
                # calculate face feature vector
                fmap = kpu.forward(task_fe, img_face)
                feature = kpu.face_encode(fmap[:])
                scores = []
                for j in range(len(record_ftrs)):
                    score = kpu.face_compare(record_ftrs[j], feature)
                    scores.append(score)
                max_score = 0
                index = 0
                for k in range(len(scores)):
                    if max_score < scores[k]:
                        max_score = scores[k]
                        index = k
                if max_score > ACCURACY:
                    a = img.draw_string(i.x()-30, i.y(), ("%s :%2.1f" % (names[index], max_score)), color=(0, 255, 0), scale=2)
                    if flag == -1:
                        flag = index
                    if flag == index:
                        record_1+=1
                    else:
                        record_1-=1
                        if record_1<0:
                            flag = -1
                    if record_1 == 8:
                        uart.write('Y')
                        record_1 = 0
                        flag = -1
                        record_2 = 0
                        key_left = False

                else:
                    a = img.draw_string(i.x()-20, i.y(), ("Stranger :%2.1f" % (max_score)), color=(255, 0, 0), scale=2)
                    record_2+=1
                    if record_2 == 16:
                        uart.write('N')
                        record_1 = 0
                        flag = -1
                        record_2 = 0
                        key_left = False
        fps = clock.fps()
       # print("%2.1f fps" % fps)
        a = lcd.display(img)
        gc.collect()

    elif start_processing:
            code = kpu.run_yolo2(task_fd, img)
            if code:
                for i in code:
                    # Cut face and resize to 128x128
                    a = img.draw_rectangle(i.rect())
                    face_cut = img.cut(i.x(), i.y(), i.w(), i.h())
                    face_cut_128 = face_cut.resize(128, 128)
                    a = face_cut_128.pix_to_ai()
                    # a = img.draw_image(face_cut_128, (0,0))
                    # Landmark for face 5 points
                    fmap = kpu.forward(task_ld, face_cut_128)
                    plist = fmap[:]
                    le = (i.x() + int(plist[0] * i.w() - 10), i.y() + int(plist[1] * i.h()))
                    re = (i.x() + int(plist[2] * i.w()), i.y() + int(plist[3] * i.h()))
                    nose = (i.x() + int(plist[4] * i.w()), i.y() + int(plist[5] * i.h()))
                    lm = (i.x() + int(plist[6] * i.w()), i.y() + int(plist[7] * i.h()))
                    rm = (i.x() + int(plist[8] * i.w()), i.y() + int(plist[9] * i.h()))
                    a = img.draw_circle(le[0], le[1], 4)
                    a = img.draw_circle(re[0], re[1], 4)
                    a = img.draw_circle(nose[0], nose[1], 4)
                    a = img.draw_circle(lm[0], lm[1], 4)
                    a = img.draw_circle(rm[0], rm[1], 4)
                    # align face to standard position
                    src_point = [le, re, nose, lm, rm]
                    T = image.get_affine_transform(src_point, dst_point)
                    a = image.warp_affine_ai(img, img_face, T)
                    a = img_face.ai_to_pix()
                    # a = img.draw_image(img_face, (128,0))
                    del (face_cut_128)
                    # calculate face feature vector
                    fmap = kpu.forward(task_fe, img_face)
                    feature = kpu.face_encode(fmap[:])
                    record_ftr = feature
                    record_ftrs.append(record_ftr)
                    save_feature(record_ftr) #存到SD卡
                    start_processing = False
                    fps = clock.fps()
                   # print("%2.1f fps" % fps)
                    a = lcd.display(img)
                    gc.collect()

    else:
        lcd.display(img)
       # print("here")
    print(" %d " %key_left)





