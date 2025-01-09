# import ffmpeg
import ffmpeg
# print(ffmpeg.__version__)

# Đọc và xoay video (xoay 90 độ)
input_file = "video_input.mp4"
rotated_output = r"D:\KhoaLuan\data_video\6026791583920_rotated.mp4"

ffmpeg.input(input_file).filter('transpose', 1).output(rotated_output).run()

# Cắt video: cắt 125px từ bên trái và 155px từ bên phải
left_margin = 125
right_margin = 155
cropped_output = r"D:\KhoaLuan\data_video\video_cropped_1.mp4"

ffmpeg.input(input_file).crop(x=left_margin, y=0, w=1920-left_margin-right_margin, h=1080).output(cropped_output).run()
