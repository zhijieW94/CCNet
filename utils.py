import os, yaml, time
from glob import glob
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

#load configs from yaml file
def get_config(config):
    with open(config,'r') as stream:
        return yaml.full_load(stream)
        # return yaml.load(stream)

# save configuration file
def write_config(config, outfile):
    with open(outfile,'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def get_current_time():
    now = int(round(time.time() * 1000))
    current_time = time.strftime('%H%M%S', time.localtime(now / 1000))
    current_time_day = time.strftime('%Y_%m_%d', time.localtime(now / 1000))
    return current_time,current_time_day

def mkdir_output_train(args):
    name = args['name']
    dir_out_base = args['dir_out']
    dir_log = args['output']['dir_log']
    dir_config = args['output']['dir_config']
    dir_sample = args['output']['dir_sample']
    dir_checkpoint = args['output']['dir_checkpoint']
    GPU_ID = args['GPU_ID']
    gpu_id = ''
    for id in GPU_ID:
        gpu_id = gpu_id + str(id)

    current_time, current_time_day = get_current_time()
    dir_out_base = os.path.join(dir_out_base, current_time_day, name, str(gpu_id)+'_'+current_time)

    dir_log = os.path.join(dir_out_base, dir_log)
    dir_config = os.path.join(dir_out_base, dir_config)
    dir_sample = os.path.join(dir_out_base, dir_sample)
    dir_checkpoint = os.path.join(dir_out_base, dir_checkpoint)
    path_record = os.path.join(dir_out_base, 'record.csv')

    check_folder(dir_log)
    check_folder(dir_config)
    check_folder(dir_sample)
    check_folder(dir_checkpoint)
    os.mknod(path_record)

    os.mknod(os.path.join(dir_config, 'grad_norm.npy'))


    write_config(args, os.path.join(dir_config, 'configs.yaml'))
    return dir_log, dir_config, dir_sample, dir_checkpoint, path_record

def mkdir_output_test(args):
    dir_out_base = args['dir_out']
    dir_result = args['output']['dir_result']
    GPU_ID = args['GPU_ID']
    name = args['name']
    phase = args['phase']
    gpu_id = ''
    for id in GPU_ID:
        gpu_id = gpu_id + str(id)

    current_time, current_time_day = get_current_time()
    # dir_out_base = os.path.join(dir_out_base, phase, current_time_day, name, str(gpu_id) + '_' + current_time)
    dir_out_base = os.path.join(dir_out_base, phase, current_time_day, name)

    dir_result = os.path.join(dir_out_base, dir_result)
    check_folder(dir_result)
    return dir_result

def get_file_list(path_file):
    list_file = []
    if os.path.isdir(path_file):
        list_file = glob(path_file + '/*.*')
    else:
        list_file.append(path_file)
    return list_file

def show_variables():
    t_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(t_vars, print_info=True)

    vars = [var for var in t_vars if 'encoder' in var.name]
    slim.model_analyzer.analyze_vars(vars, print_info=True)

    vars = [var for var in t_vars if 'attn' in var.name]
    slim.model_analyzer.analyze_vars(vars, print_info=True)

    vars = [var for var in t_vars if 'decoder' in var.name]
    slim.model_analyzer.analyze_vars(vars, print_info=True)

    vars = [var for var in t_vars if 'loss' in var.name]
    slim.model_analyzer.analyze_vars(vars, print_info=True)

def create_html_tabel(file_path,columns):
    file = open(file_path, 'w')
    file.write("<html><body><table><tr>")
    for col in columns:
        file.write("<th>%s</th>"%(col))
    file.write("</tr>")
    return file

def get_prefix(file_path):
    prefix, _ = os.path.splitext(file_path)
    prefix = os.path.basename(prefix)
    return prefix

def getContentByMask(content,mask):
    x = content.copy()
    x[np.where(mask < 1)] = 0
    return x


