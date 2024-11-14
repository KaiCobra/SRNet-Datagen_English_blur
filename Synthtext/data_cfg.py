"""
Some configurations.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License 
Written by Yu Qian
"""
import numpy as np

# font
font_size = [64, 64]
underline_rate = 0
strong_rate = 0
oblique_rate = 0
strength_rate = 0
font_dir = 'datasets/fonts/english_ttf'
standard_font_path = 'datasets/fonts/english_ttf/arial.ttf'

# text
text_filepath = 'data/texts.txt'
capitalize_rate = 0
uppercase_rate = 0

# background
bg_filepath = 'datasets/imnames.cp'
temp_bg_path = 'datasets/bg_data/bg_img/'

## background augment
brightness_rate = 0.8
brightness_min = 0.7
brightness_max = 1.5
color_rate = 0.8
color_min =0.7
color_max = 1.3
contrast_rate = 0.8
contrast_min = 0.7
contrast_max = 1.3

# curve
is_curve_rate = 0 #0.05
curve_rate_param = [0.1, 0.5] # scale, shift for np.random.randn()

# perspective
rotate_param = [-5, 5] # scale, shift for np.random.randn() [20, 0]
focal_param = [200, 600] # scale, shift for np.random.randn() [0.0, 1]
shear_param = [0, 0] # scale, shift for np.random.randn() [10, 0]
perspect_param = [0.002, 0.000] # scale, shift for np.random.randn() [0.002, 0.000]

# render

## surf augment
elastic_rate = 0.0000001 #0.0003
elastic_grid_size = 4
elastic_magnitude = 2

## colorize
# padding_ud = [5, 50]
# padding_lr = [5, 50]
padding_ud = [0, 5]
padding_lr = [0,5]
is_border_rate = 0.05
is_shadow_rate = 0
shadow_angle_degree = [1, 3, 5, 7] # shift for shadow_angle_param
shadow_angle_param = [0.5, None] # scale, shift for np.random.randn()
shadow_shift_param = np.array([[0, 1, 3], [2, 7, 15]], dtype = np.float32) # scale, shift for np.random.randn()
shadow_opacity_param = [0.1, 0.5] # shift for shadow_angle_param
color_filepath = 'data/colors.cp'
use_random_color_rate = 0.5
