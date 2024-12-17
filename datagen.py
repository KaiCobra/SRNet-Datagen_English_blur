import os
import cv2
import cfg
from Synthtext.gen import datagen, multiprocess_datagen
import time

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(data_dir: str, sample_num: int) -> None:
    
    # i_t_dir = os.path.join(cfg.data_dir, cfg.i_t_dir)
    i_s_dir = os.path.join(data_dir, cfg.i_s_dir)
    # i_s_bbox_dir = os.path.join(cfg.data_dir, cfg.i_s_bbox_dir)
    # t_sk_dir = os.path.join(cfg.data_dir, cfg.t_sk_dir)
    t_t_dir = os.path.join(data_dir, cfg.t_t_dir)
    t_b_dir = os.path.join(data_dir, cfg.t_b_dir)
    t_f_dir = os.path.join(data_dir, cfg.t_f_dir)
    # t_f_bbox_dir = os.path.join(cfg.data_dir, cfg.t_f_bbox_dir)
    mask_s_dir = os.path.join(data_dir, cfg.mask_s_dir)
    mask_t_dir = os.path.join(data_dir, cfg.mask_t_dir)
    # mask_ts_dir = os.path.join(cfg.data_dir, cfg.mask_ts_dir)
    i_s_text_dir = os.path.join(data_dir, 'i_s.txt')
    i_t_text_dir = os.path.join(data_dir, 'i_t.txt')
    # i_ts_dir = os.path.join(cfg.data_dir, cfg.i_ts_dir)
    font_dir = os.path.join(data_dir, 'font.txt')
    
    # makedirs(i_t_dir)
    makedirs(i_s_dir)
    # makedirs(t_sk_dir)
    makedirs(t_t_dir)
    makedirs(t_b_dir)
    makedirs(t_f_dir)
    makedirs(mask_s_dir)
    makedirs(mask_t_dir)
    # makedirs(mask_ts_dir)
    # makedirs(i_s_bbox_dir)
    # makedirs(t_f_bbox_dir)
    # makedirs(i_ts_dir)

    mp_gen = multiprocess_datagen(cfg.process_num, cfg.data_capacity)
    mp_gen.multiprocess_runningqueue()
    gen = datagen()
    digit_num = len(str(sample_num)) - 1

    f1 = open(i_s_text_dir,'w+')
    f2 = open(i_t_text_dir,'w+')
    f3 = open(os.path.join(data_dir,'angle_stat.txt'),'w+')
    font_file = open(font_dir,'w+')

    start_time = time.time()
    for idx in range(sample_num):
        if (idx + 1) % 50 == 0:
            end_time = time.time()
            print ("Generating step {:>6d} / {:>6d} spend time:{}".format(idx + 1, sample_num,str(end_time-start_time)))
            start_time = time.time()
        # i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = si_gen.gen_srnet_data_with_background()
        # i_t, i_s, t_sk, t_t,t_t2, t_b, t_f, i_m, mask_t,final_surf_list, text1, text2,textn,bbox1,bbox2,angle= mp_gen.dequeue_data()
        i_s, t_sk, t_t, t_b, t_f, mask_s, mask_t,final_surf_list, text1, text2,textn,bbox1,bbox2,angle,font_name= gen.gen_srnet_data_with_background()
        # i_t, i_s, t_sk, t_b, t_f, i_m, mask_t,final_surf_list, text1, text2,textn,angle= mp_gen.dequeue_data()
        
        # i_t_path = os.path.join(i_t_dir, str(idx).zfill(digit_num) + '.png')
        i_s_path = os.path.join(i_s_dir, str(idx).zfill(digit_num) + '.png')
        # t_sk_path = os.path.join(t_sk_dir, str(idx).zfill(digit_num) + '.png')
        t_t_path = os.path.join(t_t_dir, str(idx).zfill(digit_num) + '.png')
        t_b_path = os.path.join(t_b_dir, str(idx).zfill(digit_num) + '.png')
        t_f_path = os.path.join(t_f_dir, str(idx).zfill(digit_num) + '.png')
        mask_s_path = os.path.join(mask_s_dir, str(idx).zfill(digit_num) + '.png')
        # i_ts_path = os.path.join(i_ts_dir, str(idx).zfill(digit_num) + '.png')
        mask_t_path = os.path.join(data_dir, cfg.mask_t_dir, str(idx).zfill(digit_num) + '.png')


        # cv2.imwrite(i_t_path, i_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # i_s_bbox_path = os.path.join(i_s_bbox_dir, str(idx).zfill(digit_num) + '.txt')
        # text_bbox1,char_bbox1=bbox1  
              
        # with open(i_s_bbox_path,'w',encoding='utf-8') as bbox_fw:
        #     flat_list = []
        #     for sublist in text_bbox1:
        #         for item in sublist:
        #             flat_list.append(str(item))
        #     text_bbox_str = ' '.join(flat_list)
        #     bbox_fw.writelines(text1+' '+text_bbox_str+'\n')
        #     for char1,char_bbox in zip(text1,char_bbox1):
        #         char_flat_list = []
        #         for char_sublist in char_bbox:
        #             for char_item in char_sublist:
        #                 char_flat_list.append(str(char_item))
        #         char_bbox_str = ' '.join(char_flat_list)
        #         bbox_fw.writelines(char1+' '+char_bbox_str+'\n')

        # t_f_bbox_path = os.path.join(t_f_bbox_dir, str(idx).zfill(digit_num) + '.txt')
        # text_bbox2,char_bbox2=bbox2
        
        # with open(t_f_bbox_path,'w',encoding='utf-8') as bbox_fw:
        #     flat_list = []
        #     for sublist in text_bbox2:
        #         for item in sublist:
        #             flat_list.append(str(item))
        #     text_bbox_str = ' '.join(flat_list)
        #     bbox_fw.writelines(text2+' '+text_bbox_str+'\n')
        #     for char2,char_bbox in zip(text2,char_bbox2):
        #         char_flat_list = []
        #         for char_sublist in char_bbox:
        #             for char_item in char_sublist:
        #                 char_flat_list.append(str(char_item))
        #         char_bbox_str = ' '.join(char_flat_list)
        #         bbox_fw.writelines(char2+' '+char_bbox_str+'\n')
        # for i in char_bbox1:
        #     lt, rt, rb, lb = i
        #     lt, rt, rb, lb = tuple(lt), tuple(rt), tuple(rb), tuple(lb)
        #     cv2.line(i_s, lt, rt, 127, 2)
        #     cv2.line(i_s, rt, rb, 127, 2)
        #     cv2.line(i_s, rb, lb, 127, 2)
        #     cv2.line(i_s, lb, lt, 127, 2)
        cv2.imwrite(i_s_path, i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # cv2.imwrite(t_sk_path, t_sk, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_t_path, t_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_b_path, t_b, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_f_path, t_f, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(mask_s_path, mask_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(mask_t_path, mask_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        ###generate mask_ts####
        # mask_ts_path = os.path.join(cfg.data_dir, cfg.mask_ts_dir, str(idx).zfill(digit_num))
        # makedirs(mask_ts_path)
        # f_n = open(os.path.join(mask_ts_path, 'gt_n.txt'),'w+')
        # for num_ts,mask_ts in enumerate(final_surf_list):
        #     f_n.writelines(str(num_ts)+'.png '+textn[num_ts]+'\n') 
           
        #     mask_ts_img_path = os.path.join(mask_ts_path,(str(num_ts)+'.png'))
        #     cv2.imwrite(mask_ts_img_path, mask_ts, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        ####generate mask_ts####
            
        # cv2.imwrite(i_ts_path, i_ts, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        f1.writelines(str(idx).zfill(digit_num) + '.png '+text1+'\n')
        f2.writelines(str(idx).zfill(digit_num) + '.png '+text2+'\n')
        angle = [str(angle1-200) if angle1==angle[-1] else str(angle1+60) for angle1 in angle]
        stat = ' '.join(angle)
        f3.writelines(str(idx).zfill(digit_num) + '.png '+stat+'\n')
        font_file.writelines(str(idx).zfill(digit_num) + '.png '+ font_name+'\n')


        # print(angle)
        #print(idx)

    
    mp_gen.terminate_pool()
    f3.close()
    f1.close()
    f2.close()
    font_file.close()

if __name__ == '__main__':
    main(data_dir = cfg.data_dir, sample_num = cfg.sample_num)
